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

# LiDAR-based soft safety (front sector only)
USE_LIDAR = True
SOFT_STOP_MARGIN = 0.10   # keep at least 10 cm from any front obstacle
FRONT_SECTOR_DEG = 60.0   # ±30° front cone


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

        # LiDAR (optional but recommended)
        self.lidar = None
        if USE_LIDAR:
            try:
                self.lidar = self.robot.getDevice("LDS-01")
                self.lidar.enable(self.timestep)
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
            # For versions where DetectorParameters_create does not exist
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
    def get_best_marker_pose(self, allowed_ids=None):
        """
        Detect ArUco markers and return pose of the closest marker whose ID is allowed.

        Parameters
        ----------
        allowed_ids : iterable of int or None
            If not None, restrict detection to this list of ArUco IDs. If None,
            use the global TARGET_IDS list.

        Returns
        -------
        (found, (X, Y, Z), marker_id)
            Pose in camera frame and the corresponding marker ID.
            Z is forward, X is right, Y is down.
        """
        if not HAVE_CV or self.aruco_dict is None or self.cam_matrix is None:
            return False, None, None

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        img_bytes = self.camera.getImage()
        if img_bytes is None:
            return False, None, None

        # Webots camera returns BGRA
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return False, None, None

        ids = ids.flatten()

        # Decide which IDs are acceptable for this call
        if allowed_ids is None:
            allowed_ids = TARGET_IDS

        if not allowed_ids:
            return False, None, None

        allowed_set = {int(a) for a in allowed_ids}

        # Filter out only markers whose ID is in allowed_ids
        valid_indices = [i for i, mid in enumerate(ids) if int(mid) in allowed_set]
        if not valid_indices:
            return False, None, None

        # Pose estimation for all detected markers
        # estimatePoseSingleMarkers expects the whole list
        rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, self.cam_matrix, self.dist_coeffs
        )

        # Pick the closest valid target (smallest Z)
        best_idx = None
        best_Z = float("inf")
        for idx in valid_indices:
            Z = float(tvecs[idx][0][2])  # tvecs[idx] is shape (1,3)
            if Z < best_Z:
                best_Z = Z
                best_idx = idx

        if best_idx is None:
            return False, None, None

        tvec = tvecs[best_idx][0]  # [X, Y, Z]
        X = float(tvec[0])
        Y = float(tvec[1])
        Z = float(tvec[2])
        marker_id = int(ids[best_idx])

        return True, (X, Y, Z), marker_id

    # ------------------------------------------------------------------ #
    def pbvs_control(self, X, Z):
        """
        Simple PBVS control law for mobile robot:
        - Use (Z - DESIRED_Z) to set forward speed
        - Use bearing atan2(X, Z) to set angular speed
        """
        # If we are behind desired distance, no forward motion
        if Z <= 0.0:
            return 0.0, 0.0

        # Bearing angle of marker in camera frame
        bearing = math.atan2(X, Z)  # rad; +ve means marker to the right

        # Distance error in depth
        z_err = Z - DESIRED_Z

        # Control law
        omega = -K_YAW * bearing          # negative to turn towards marker
        v = KV_Z * z_err * math.cos(bearing)

        # Limit speeds
        v = clamp(v, -MAX_V, MAX_V)
        omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)

        # When bearing is large, prioritize turning
        if abs(bearing) > 0.5:   # ~30 degrees
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
                # Nothing we can do without vision
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            v_cmd = 0.0
            omega_cmd = 0.0

            # 1) Get marker pose in camera frame for the CURRENT target only
            if self.goal_reached or self.current_target_id is None:
                found = False
                pose = None
                marker_id = None
            else:
                found, pose, marker_id = self.get_best_marker_pose([self.current_target_id])

            if found and not self.goal_reached:
                X, Y, Z = pose

                # Extra safety: verify ID, even though we filtered by it
                if marker_id != self.current_target_id:
                    print(f"[DEBUG] Saw marker {marker_id}, ignoring; expecting {self.current_target_id}")
                else:
                    # Check if we are “close enough” to the desired pose for this marker
                    if (abs(Z - DESIRED_Z) <= Z_TOL):
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
                        # PBVS control towards the current marker
                        v_cmd, omega_cmd = self.pbvs_control(X, Z)

            if not found:
                # Marker not visible or all goals reached -> stop or slowly search
                if not self.goal_reached and self.current_target_id is not None:
                    v_cmd = 0.0
                    omega_cmd = 0.4   # slow rotation to find the marker
                else:
                    v_cmd = 0.0
                    omega_cmd = 0.0

            # 2) LiDAR soft limiter (only shrink v, never flip direction or spin)
            front_dist = self.get_front_distance() if USE_LIDAR else float("inf")
            if math.isfinite(front_dist):
                # Available safe distance ahead (minus margin)
                safe_space = front_dist - SOFT_STOP_MARGIN
                if safe_space <= 0.0 and v_cmd > 0.0:
                    v_cmd = 0.0
                elif v_cmd > 0.0:
                    # Limit speed so we do not try to run faster than remaining space
                    v_cmd = min(v_cmd, safe_space / 0.5)  # simple scaling factor

            # 3) Apply differential drive inverse and clamp wheel speeds
            v_l, v_r = diff_drive_inverse(v_cmd, omega_cmd)
            v_l = clamp(v_l, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
            v_r = clamp(v_r, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

            self.left_motor.setVelocity(v_l)
            self.right_motor.setVelocity(v_r)


if __name__ == "__main__":
    controller = PBVSTurtleBot()
    controller.run()
