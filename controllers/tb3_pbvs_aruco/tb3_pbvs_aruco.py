from controller import Robot
import math

# --- OpenCV + NumPy ---------------------------------------------------------
try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False
    print("[ERROR] OpenCV + numpy not available in this Python. "
          "Install opencv-contrib-python and numpy for PBVS to work.")


# =========================== PARAMETERS =====================================

# ArUco markers to visit in order
TARGET_IDS = [1, 2, 3, 4, 5, 6]

# ArUco dictionary
ARUCO_DICT_NAME = "DICT_4X4_50"

# Marker side length in meters (must match Webots world)
MARKER_LENGTH = 0.14

# Desired marker pose in camera frame
DESIRED_Z = 0.20          # 15 cm in front of camera
DESIRED_X = 0.0           # centered horizontally

# Goal tolerances (0 means exact match)
Z_TOL = 0.03
X_TOL = 0.03

# TurtleBot3 kinematics
WHEEL_RADIUS = 0.033      # m
AXLE_LENGTH = 0.16        # m
MAX_WHEEL_VEL = 6.5       # rad/s

# PBVS gains
KV_Z = 1.0                # forward gain
K_YAW = 2.0               # yaw gain

MAX_V = 0.25              # m/s
MAX_OMEGA = 1.5           # rad/s

# LiDAR safety
USE_LIDAR = True
SOFT_STOP_MARGIN = 0.10   # m
FRONT_SECTOR_DEG = 60.0   # deg (±30°)

# Obstacle avoidance
OBSTACLE_DIST = 0.35        # m, start steering
AVOID_FORWARD_SPEED = 0.12  # m/s while avoiding
AVOID_TURN_SPEED = 0.8      # rad/s additional turn


# ========================== UTILITY FUNCS ===================================

def diff_drive_inverse(v, omega):
    """(v, omega) -> (left_w, right_w)."""
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
            m.setPosition(float('inf'))
            m.setVelocity(0.0)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # LiDAR
        try:
            self.lidar = self.robot.getDevice("LDS-01")
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
        except Exception:
            print("[WARN] LiDAR 'LDS-01' not found; safety limiting disabled.")
            self.lidar = None

        # ArUco + intrinsics
        self.aruco_dict = None
        self.aruco_params = None
        self.cam_matrix = None
        self.dist_coeffs = None

        if HAVE_CV:
            self._init_aruco()
            self._init_camera_intrinsics()
        else:
            print("[ERROR] Cannot run PBVS without OpenCV + numpy.")

        # Multi-target state
        self.target_ids = TARGET_IDS[:]
        self.current_target_idx = 0
        self.all_targets_done = False

        if len(self.target_ids) == 0:
            self.all_targets_done = True
            print("[WARN] No TARGET_IDS defined. Robot will stay still.")

    def _current_target_id(self):
        if self.all_targets_done:
            return None
        return self.target_ids[self.current_target_idx]

    def _advance_to_next_target(self):
        self.current_target_idx += 1
        if self.current_target_idx >= len(self.target_ids):
            self.all_targets_done = True
            print("[INFO] All markers reached. Stopping at last marker.")
        else:
            next_id = self._current_target_id()
            print(f"[INFO] Now heading to next marker ID {next_id}.")

    def _init_aruco(self):
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, ARUCO_DICT_NAME)
            )
        except AttributeError:
            print(f"[ERROR] ArUco dictionary {ARUCO_DICT_NAME} not available.")
            self.aruco_dict = None
            return

        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        print("[INFO] ArUco detector initialized with", ARUCO_DICT_NAME)

    def _init_camera_intrinsics(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        fov = self.camera.getFov()  # rad (horizontal)

        fx = width / (2.0 * math.tan(fov / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        self.cam_matrix = np.array([[fx,  0, cx],
                                    [0,  fy, cy],
                                    [0,   0,  1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        print("[INFO] Camera intrinsics matrix:")
        print(self.cam_matrix)

    def get_lidar_sector_mins(self):
        """Return (left_min, front_min, right_min) distances."""
        if self.lidar is None:
            return float('inf'), float('inf'), float('inf')

        ranges = self.lidar.getRangeImage()
        res = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()

        if res <= 1 or fov <= 0.0:
            return float('inf'), float('inf'), float('inf')

        half_front = math.radians(FRONT_SECTOR_DEG / 2.0)
        left_min = float('inf')
        front_min = float('inf')
        right_min = float('inf')

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            angle = -fov / 2.0 + i * fov / (res - 1)

            if -half_front <= angle <= half_front:
                if r < front_min:
                    front_min = r
            elif angle < -half_front:
                if r < left_min:
                    left_min = r
            else:
                if r < right_min:
                    right_min = r

        return left_min, front_min, right_min

    def get_best_marker_pose(self, desired_id=None):
        """Return (found, (X,Y,Z)) for closest marker with given ID."""
        if not HAVE_CV or self.aruco_dict is None or self.cam_matrix is None:
            return False, None

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        img_bytes = self.camera.getImage()
        if img_bytes is None:
            return False, None

        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return False, None

        ids = ids.flatten()
        if desired_id is None:
            return False, None

        valid_indices = [i for i, mid in enumerate(ids) if mid == desired_id]
        if not valid_indices:
            return False, None

        rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, self.cam_matrix, self.dist_coeffs
        )

        best_idx = None
        best_Z = float('inf')
        for idx in valid_indices:
            Z = float(tvecs[idx][0][2])
            if Z < best_Z:
                best_Z = Z
                best_idx = idx

        if best_idx is None:
            return False, None

        tvec = tvecs[best_idx][0]
        X = float(tvec[0])
        Y = float(tvec[1])
        Z = float(tvec[2])

        return True, (X, Y, Z)

    def pbvs_control(self, X, Z):
        """PBVS law -> (v, omega)."""
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

    def step(self):
        return self.robot.step(self.timestep)

    def run(self):
        print("[INFO] PBVS TurtleBot controller started.")
        print("[INFO] Will visit markers in order:", self.target_ids)
        if not self.all_targets_done:
            print(f"[INFO] Starting with marker ID {self._current_target_id()}")
        print("[INFO] Desired camera-frame pose: X=%.2f, Z=%.2f" %
              (DESIRED_X, DESIRED_Z))

        while self.step() != -1:
            if not HAVE_CV or self.cam_matrix is None:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            if self.all_targets_done:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            current_id = self._current_target_id()
            v_cmd = 0.0
            omega_cmd = 0.0

            # PBVS towards current marker
            found, pose = self.get_best_marker_pose(desired_id=current_id)

            if found:
                X, Y, Z = pose

                if (abs(X - DESIRED_X) < X_TOL) and (abs(Z - DESIRED_Z) < Z_TOL):
                    print(f"[INFO] Reached marker {current_id} "
                          f"(X={X:.2f}, Z={Z:.2f}).")
                    self._advance_to_next_target()
                    v_cmd = 0.0
                    omega_cmd = 0.0
                else:
                    v_cmd, omega_cmd = self.pbvs_control(X, Z)
            else:
                # Search spin
                v_cmd = 0.0
                omega_cmd = 0.4

            # LiDAR-based obstacle avoidance
            if USE_LIDAR and self.lidar is not None:
                left_d, front_d, right_d = self.get_lidar_sector_mins()

                if math.isfinite(front_d) and front_d < OBSTACLE_DIST and v_cmd > 0.0:
                    v_cmd = min(v_cmd, AVOID_FORWARD_SPEED)

                    if not math.isfinite(left_d):
                        left_d = float('inf')
                    if not math.isfinite(right_d):
                        right_d = float('inf')

                    if left_d > right_d:
                        omega_cmd += AVOID_TURN_SPEED
                    else:
                        omega_cmd -= AVOID_TURN_SPEED

                    omega_cmd = clamp(omega_cmd, -MAX_OMEGA, MAX_OMEGA)

                front_dist = front_d
                if math.isfinite(front_dist):
                    safe_space = front_dist - SOFT_STOP_MARGIN
                    if safe_space <= 0.0 and v_cmd > 0.0:
                        v_cmd = 0.0
                    elif v_cmd > 0.0:
                        v_cmd = min(v_cmd, safe_space / 0.5)

            # Apply wheel speeds
            v_l, v_r = diff_drive_inverse(v_cmd, omega_cmd)
            v_l = clamp(v_l, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            v_r = clamp(v_r, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

            self.left_motor.setVelocity(v_l)
            self.right_motor.setVelocity(v_r)


if __name__ == "__main__":
    controller = PBVSTurtleBot()
    controller.run()