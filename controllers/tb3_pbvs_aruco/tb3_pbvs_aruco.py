from controller import Robot
import math

# --- Try to import OpenCV + numpy -------------------------------------------
try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False
    print("[ERROR] OpenCV + numpy not available in this Python. "
          "Install opencv-contrib-python and numpy for PBVS to work.")


# =========================== PARAMETERS =====================================

# ArUco marker(s) that represent the goal
TARGET_IDS = [1, 2, 3, 4, 5]          # you can put [1, 5, 7] etc; we will visit them in order

# ArUco dictionary used when printing the markers
ARUCO_DICT_NAME = "DICT_4X4_50"

# Physical size of the marker's side in meters (VERY IMPORTANT)
MARKER_LENGTH = 0.14      # e.g. 14 cm; change to your actual printed size

# Desired final relative pose of marker in camera frame (PBVS target)
DESIRED_Z = 0.20          # want marker 20 cm in front of camera
DESIRED_X = 0.20          # centered horizontally

# Tolerances to declare "goal reached"
Z_TOL = 0.03              # 3 cm
X_TOL = 0.03              # 3 cm

# Robot kinematics (TurtleBot3 Burger approx)
WHEEL_RADIUS = 0.033      # m
AXLE_LENGTH = 0.16        # m (distance between wheels)
MAX_WHEEL_VEL = 6.5       # rad/s (slightly under spec)

# PBVS gains
KV_Z = 1.0                # gain on (Z - DESIRED_Z)  -> forward speed
K_YAW = 2.0               # gain on bearing angle atan2(X, Z) -> angular speed

MAX_V = 0.25              # max forward speed (m/s)
MAX_OMEGA = 1.5           # max angular speed (rad/s)

# LiDAR-based safety / obstacle handling
USE_LIDAR = True

# 🔧 WIDENED: use roughly half-circle in front so “half-screen” obstacles count
FRONT_SECTOR_DEG = 180.0      # ±90° front cone for front distance

# We still keep your “come close then decide marker vs obstacle” idea:
OBSTACLE_CHECK_DIST = 0.10    # come as close as 10 cm, then decide
SOFT_STOP_MARGIN = 0.02       # tiny extra safety margin when it *is* a marker


# ========================== UTILITY FUNCS ===================================

def diff_drive_inverse(v, omega):
    """
    Convert (linear v, angular omega) into left/right wheel angular velocities.
    """
    v_r = (2.0 * v + omega * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
    v_l = (2.0 * v - omega * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
    return v_l, v_r


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ========================== MAIN CONTROLLER =================================

class PBVSTurtleBot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float('inf'))  # velocity control mode
            m.setVelocity(0.0)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # LiDAR
        self.lidar = None
        if USE_LIDAR:
            try:
                self.lidar = self.robot.getDevice("LDS-01")
                self.lidar.enable(self.timestep)
                # 🔧 NEW: enable visual point-cloud so you can see it in Optional Rendering
                self.lidar.enablePointCloud()
            except Exception:
                print("[WARN] LiDAR 'LDS-01' not found; safety limiting disabled.")
                self.lidar = None

        # Prepare ArUco detector and camera intrinsics
        self.aruco_dict = None
        self.aruco_params = None
        self.cam_matrix = None
        self.dist_coeffs = None

        if HAVE_CV:
            self._init_aruco()
            self._init_camera_intrinsics()
        else:
            print("[ERROR] Cannot run PBVS without OpenCV + numpy.")

        # Multi-marker navigation state
        self.goal_reached = False   # True when all markers in TARGET_IDS are reached
        self.current_target_index = 0
        self.current_target_id = TARGET_IDS[0] if TARGET_IDS else None

    # ------------------------------------------------------------------ #
    def _init_aruco(self):
        """
        Set up ArUco dictionary and detection parameters.
        """
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, ARUCO_DICT_NAME)
            )
        except AttributeError:
            print(f"[ERROR] ArUco dictionary {ARUCO_DICT_NAME} not available.")
            self.aruco_dict = None
            return

        # Different OpenCV versions expose DetectorParameters differently
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        print("[INFO] ArUco detector initialized with", ARUCO_DICT_NAME)

    # ------------------------------------------------------------------ #
    def _init_camera_intrinsics(self):
        """
        Approximate camera intrinsics from Webots camera parameters.
        """
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        fov = self.camera.getFov()  # horizontal FOV in radians

        # Pinhole intrinsics approximation for simulation
        fx = width / (2.0 * math.tan(fov / 2.0))
        fy = fx  # assume square pixels
        cx = width / 2.0
        cy = height / 2.0

        self.cam_matrix = np.array([[fx,  0, cx],
                                    [0,  fy, cy],
                                    [0,   0,  1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        print("[INFO] Camera intrinsics matrix:")
        print(self.cam_matrix)

    # ------------------------------------------------------------------ #
    def get_front_distance(self):
        """
        Minimum distance in a front LiDAR sector (if LiDAR exists).
        Returns +inf if LiDAR missing or all invalid.

        🔧 Uses a wide ±90° sector now, so even obstacles that “cover half
        the screen” but are a bit off-center will trigger.
        """
        if self.lidar is None:
            return float('inf')

        ranges = self.lidar.getRangeImage()
        res = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()

        if res <= 1 or fov <= 0.0:
            return float('inf')

        half_sector = math.radians(FRONT_SECTOR_DEG / 2.0)
        min_front = float('inf')

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            angle = -fov / 2.0 + i * fov / (res - 1)
            if -half_sector <= angle <= half_sector:
                if r < min_front:
                    min_front = r

        return min_front

    # ------------------------------------------------------------------ #
    def compute_lidar_turn(self):
        """Decide a turn direction based on LiDAR point cloud.
        Looks at left vs right free space and returns an angular velocity
        (sign only; magnitude is modest) that steers towards the more open side.
        """
        if self.lidar is None:
            return 0.0

        ranges = self.lidar.getRangeImage()
        res = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()

        if res <= 1 or fov <= 0.0:
            return 0.0

        left_sum = 0.0
        left_cnt = 0
        right_sum = 0.0
        right_cnt = 0

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            angle = -fov / 2.0 + i * fov / (res - 1)
            # negative angle = right, positive = left
            if angle < 0.0:
                right_sum += r
                right_cnt += 1
            elif angle > 0.0:
                left_sum += r
                left_cnt += 1

        left_avg = left_sum / left_cnt if left_cnt > 0 else 0.0
        right_avg = right_sum / right_cnt if right_cnt > 0 else 0.0

        TURN_SPEED = 0.8

        if left_avg == 0.0 and right_avg == 0.0:
            return TURN_SPEED  # arbitrary

        if left_avg > right_avg:
            return TURN_SPEED      # turn left
        elif right_avg > left_avg:
            return -TURN_SPEED     # turn right
        else:
            return TURN_SPEED

    # ------------------------------------------------------------------ #
    def get_best_marker_pose(self, allowed_ids=None):
        """
        Detect ArUco markers and return pose of the closest marker whose ID is allowed.

        Returns (found, (X, Y, Z), marker_id) in camera frame.
        """
        if not HAVE_CV or self.aruco_dict is None or self.cam_matrix is None:
            return False, None, None

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        img_bytes = self.camera.getImage()
        if img_bytes is None:
            return False, None, None

        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return False, None, None

        ids = ids.flatten()

        if allowed_ids is None:
            allowed_ids = TARGET_IDS

        if not allowed_ids:
            return False, None, None

        allowed_set = {int(a) for a in allowed_ids}
        valid_indices = [i for i, mid in enumerate(ids) if int(mid) in allowed_set]
        if not valid_indices:
            return False, None, None

        rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, self.cam_matrix, self.dist_coeffs
        )

        best_idx = None
        best_Z = float("inf")
        for idx in valid_indices:
            Z = float(tvecs[idx][0][2])
            if Z < best_Z:
                best_Z = Z
                best_idx = idx

        if best_idx is None:
            return False, None, None

        tvec = tvecs[best_idx][0]
        X = float(tvec[0])
        Y = float(tvec[1])
        Z = float(tvec[2])
        marker_id = int(ids[best_idx])

        return True, (X, Y, Z), marker_id

    # ------------------------------------------------------------------ #
    def pbvs_control(self, X, Z):
        """
        PBVS control law:
        - Use (Z - DESIRED_Z) for forward speed
        - Use bearing atan2(X, Z) for angular speed
        """
        if Z <= 0.0:
            return 0.0, 0.0

        bearing = math.atan2(X, Z)
        z_err = Z - DESIRED_Z

        omega = -K_YAW * bearing
        v = KV_Z * z_err * math.cos(bearing)

        v = clamp(v, -MAX_V, MAX_V)
        omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)

        if abs(bearing) > 0.5:
            v = 0.0

        return v, omega

    # ------------------------------------------------------------------ #
    def step(self):
        return self.robot.step(self.timestep)

    # ------------------------------------------------------------------ #
    def run(self):
        print("[INFO] PBVS TurtleBot controller started.")
        print("[INFO] TARGET_IDS:", TARGET_IDS)
        print("[INFO] Desired camera-frame marker pose: X=%.2f, Z=%.2f" %
              (DESIRED_X, DESIRED_Z))

        if TARGET_IDS:
            print(f"[INFO] Marker visit sequence: {TARGET_IDS}")
            print(f"[INFO] Starting with marker {self.current_target_id}")
        else:
            print("[WARN] TARGET_IDS is empty; robot will not move toward any marker.")

        while self.step() != -1:
            if not HAVE_CV or self.cam_matrix is None:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            v_cmd = 0.0
            omega_cmd = 0.0

            # 1) Marker → marker logic (unchanged)
            if self.goal_reached or self.current_target_id is None:
                found = False
                pose = None
                marker_id = None
            else:
                found, pose, marker_id = self.get_best_marker_pose([self.current_target_id])

            if found and not self.goal_reached:
                X, Y, Z = pose

                if marker_id != self.current_target_id:
                    print(f"[DEBUG] Saw marker {marker_id}, ignoring; expecting {self.current_target_id}")
                else:
                    if abs(Z - DESIRED_Z) <= Z_TOL:
                        is_last = (self.current_target_index == len(TARGET_IDS) - 1)

                        if is_last:
                            print(f"[INFO] Reached final marker {marker_id}. "
                                  f"All markers done, holding position.")
                            self.goal_reached = True
                            v_cmd = 0.0
                            omega_cmd = 0.0
                        else:
                            next_id = TARGET_IDS[self.current_target_index + 1]
                            print(f"[INFO] Reached marker {marker_id}. "
                                  f"Moving to next marker {next_id}.")
                            self.current_target_index += 1
                            self.current_target_id = next_id
                            v_cmd = 0.0
                            omega_cmd = 0.0
                    else:
                        v_cmd, omega_cmd = self.pbvs_control(X, Z)
            else:
                if not self.goal_reached and self.current_target_id is not None:
                    v_cmd = 0.0
                    omega_cmd = 0.4   # slow rotation to find the marker
                else:
                    v_cmd = 0.0
                    omega_cmd = 0.0

            # 2) LiDAR: “come to 0.1 m, then decide marker vs obstacle”
            if USE_LIDAR:
                front_dist = self.get_front_distance()
            else:
                front_dist = float('inf')

            obstacle_is_marker = (
                found
                and not self.goal_reached
                and self.current_target_id is not None
                and marker_id == self.current_target_id
            )

            if math.isfinite(front_dist) and front_dist <= OBSTACLE_CHECK_DIST:
                # Really close to something (within 10 cm)
                if not obstacle_is_marker:
                    # Not the current marker: treat as obstacle
                    if v_cmd > 0.0:
                        v_cmd = 0.0
                    turn = self.compute_lidar_turn()
                    if abs(turn) > abs(omega_cmd):
                        omega_cmd = turn
                else:
                    # Close, but it *is* the marker: let PBVS finish, with a tiny margin
                    safe_space = front_dist - SOFT_STOP_MARGIN
                    if v_cmd > 0.0 and math.isfinite(safe_space):
                        if safe_space <= 0.0:
                            v_cmd = 0.0
                        else:
                            v_cmd = min(v_cmd, safe_space / 0.2)
            else:
                # Slightly further away: mostly leave PBVS alone, maybe creep in if tight
                if math.isfinite(front_dist) and v_cmd > 0.0:
                    safe_space = front_dist - (OBSTACLE_CHECK_DIST + SOFT_STOP_MARGIN)
                    if safe_space <= 0.0:
                        v_cmd = min(v_cmd, 0.05)

            # 3) Apply wheel commands
            v_l, v_r = diff_drive_inverse(v_cmd, omega_cmd)
            v_l = clamp(v_l, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            v_r = clamp(v_r, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

            self.left_motor.setVelocity(v_l)
            self.right_motor.setVelocity(v_r)


if __name__ == "__main__":
    controller = PBVSTurtleBot()
    controller.run()
