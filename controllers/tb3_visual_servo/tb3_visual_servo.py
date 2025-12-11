from controller import Robot
import math

# Optional: if you have OpenCV + numpy installed in the Python used by Webots
try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False
    print("[WARN] OpenCV+numpy not found; ArUco-based visual servoing will be disabled.")

# ---- User parameters --------------------------------------------------------

# ArUco path: list the marker IDs in the order you want to visit them
# Example: path  -> marker 1 -> marker 2 -> marker 3 (final goal)
PATH_MARKERS = [1, 2, 3, 4, 5]   # <-- change to the IDs you actually print & place

SAFE_LIDAR_SOFT = 0.50   # start slowing down when closer than 0.5 m
SAFE_LIDAR_HARD = 0.18   # completely stop moving forward when closer than 0.18 m

# Marker dictionary used when generating ArUco images (must match your PNGs)
ARUCO_DICT_NAME = "DICT_4X4_50"  # common small dictionary

# Linear / angular speed settings (in robot frame)
BASE_LINEAR_V = 0.15         # m/s when tracking marker
SEARCH_ANGULAR_V = 0.6       # rad/s for searching when marker lost
SAFE_LIDAR_DISTANCE = 0.4    # m: stop and rotate if anything comes closer

# Differential-drive geometry for TurtleBot3 Burger (approximate)
WHEEL_RADIUS = 0.033   # m
AXLE_LENGTH = 0.16     # m  distance between wheels

MAX_WHEEL_VEL = 6.5    # rad/s (slightly below 6.67 max)

# Visual servoing gains
KP_ANGLE = 1.0         # gain from pixel error -> angular velocity
KD_ANGLE = 0.0

# Marker "close enough" threshold in pixels (side length)
CLOSE_MARKER_PX = 180


def diff_drive_inverse(v, omega):
    """Convert (v, omega) to left/right wheel velocities (rad/s)."""
    v_r = (2.0 * v + omega * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
    v_l = (2.0 * v - omega * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
    return v_l, v_r


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class VisualServoTurtleBot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Devices: wheel motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float('inf'))   # velocity control mode
            m.setVelocity(0.0)

        # Lidar (RobotisLds01 proto normally names it "LDS-01")
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.timestep)

        # Camera (you must create it in Webots and name it exactly "camera")
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        if HAVE_CV:
            # Prepare ArUco detector
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, ARUCO_DICT_NAME)
            )
            # OpenCV versions differ: some have DetectorParameters_create(), some only DetectorParameters
            if hasattr(cv2.aruco, "DetectorParameters_create"):
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            else:
                self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_dict = None
            self.aruco_params = None
        # State for path following
        self.current_target_index = 0
        self.goal_reached = False
        self.last_angle_err = 0.0

    def get_front_obstacle_distance(self, sector_degrees=60.0):
        """
        Returns the minimum LiDAR distance in a front sector (±sector_degrees/2 around 0°).
        If nothing valid, returns +inf.
        """
        ranges = self.lidar.getRangeImage()
        res = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()  # radians

        if res <= 1 or fov <= 0.0:
            return float('inf')

        half_sector = math.radians(sector_degrees / 2.0)
        min_front = float('inf')

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            # angle of this beam relative to the robot's forward direction
            angle = -fov / 2.0 + i * fov / (res - 1)
            if -half_sector <= angle <= half_sector:
                if r < min_front:
                    min_front = r

        return min_front

    def get_lidar_min_distance(self):
        ranges = self.lidar.getRangeImage()
        finite_ranges = [r for r in ranges if math.isfinite(r)]
        if not finite_ranges:
            return float('inf')
        return min(finite_ranges)

    def compute_visual_servo(self):
        """
        Use camera + ArUco to compute desired (v, omega).
        Returns (has_marker, v, omega, close_to_marker)
        """
        if not HAVE_CV or self.goal_reached or self.current_target_index >= len(PATH_MARKERS):
            return False, 0.0, 0.0, False

        # Grab image from Webots camera
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        img_bytes = self.camera.getImage()
        if img_bytes is None:
            return False, 0.0, 0.0, False

        # BGRA -> numpy array
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # ArUco detection
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                  parameters=self.aruco_params)
        if ids is None or len(ids) == 0:
            return False, 0.0, 0.0, False

        ids = ids.flatten()
        target_id = PATH_MARKERS[self.current_target_index]

        # Find first marker with the target ID
        target_idx = None
        for i, marker_id in enumerate(ids):
            if marker_id == target_id:
                target_idx = i
                break

        if target_idx is None:
            # target marker not visible yet
            return False, 0.0, 0.0, False

        pts = corners[target_idx][0]  # 4x2 array
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()

        # Horizontal pixel error (center of image is desired)
        err_x = cx - (width / 2.0)

        # PD on yaw error
        angle_err = err_x / (width / 2.0)  # normalized [-1, 1] approx
        d_err = angle_err - self.last_angle_err
        self.last_angle_err = angle_err

        omega = - (KP_ANGLE * angle_err + KD_ANGLE * d_err)
        v = BASE_LINEAR_V * max(0.0, 1.0 - abs(angle_err))  # slower if large error

        # Estimate marker apparent size (use diagonal)
        side1 = np.linalg.norm(pts[0] - pts[1])
        side2 = np.linalg.norm(pts[1] - pts[2])
        marker_size_px = 0.5 * (side1 + side2)
        close = marker_size_px >= CLOSE_MARKER_PX

        return True, v, omega, close

    def step(self):
        return self.robot.step(self.timestep)

    def run(self):
        print("Visual servoing TurtleBot3 controller started.")
        print("Path marker IDs:", PATH_MARKERS)

        while self.step() != -1:
            v = 0.0
            omega = 0.0

            # 1) Visual servoing towards current ArUco marker
            has_marker, v_vs, omega_vs, close = self.compute_visual_servo()

            if has_marker:
                v = v_vs
                omega = omega_vs

                # Check if we reached this marker, then switch to next
                if close and not self.goal_reached:
                    print(f"Reached marker ID {PATH_MARKERS[self.current_target_index]}")
                    self.current_target_index += 1
                    if self.current_target_index >= len(PATH_MARKERS):
                        print("All markers reached. Goal reached.")
                        self.goal_reached = True
            else:
                # If no marker: slowly rotate to search
                if not self.goal_reached:
                    v = 0.0
                    omega = SEARCH_ANGULAR_V

            # 2) LiDAR-based safety override (only front sector, smooth slowdown)
            front_dist = self.get_front_obstacle_distance(sector_degrees=60.0)

            if math.isfinite(front_dist):
                if front_dist < SAFE_LIDAR_SOFT:
                    # Compute slowdown factor between 0 and 1
                    # front_dist >= SAFE_LIDAR_SOFT  -> factor = 1  (no slowdown)
                    # front_dist <= SAFE_LIDAR_HARD  -> factor = 0  (full stop)
                    if front_dist <= SAFE_LIDAR_HARD:
                        slow_factor = 0.0
                    else:
                        slow_factor = (front_dist - SAFE_LIDAR_HARD) / (SAFE_LIDAR_SOFT - SAFE_LIDAR_HARD)
                        slow_factor = max(0.0, min(1.0, slow_factor))

                    # Apply slowdown to linear velocity only
                    old_v = v
                    v = v * slow_factor

                    if slow_factor == 0.0 and old_v > 0.0:
                        # We were moving forward and have to stop => optionally rotate a bit to search
                        print(f"[LIDAR] Hard stop at {front_dist:.2f} m")
                        if not self.goal_reached and not has_marker:
                            omega = SEARCH_ANGULAR_V


            # 3) Convert (v, omega) to wheel speeds and apply
            v_l, v_r = diff_drive_inverse(v, omega)
            v_l = clamp(v_l, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            v_r = clamp(v_r, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

            self.left_motor.setVelocity(v_l)
            self.right_motor.setVelocity(v_r)


if __name__ == "__main__":
    controller = VisualServoTurtleBot()
    controller.run()
