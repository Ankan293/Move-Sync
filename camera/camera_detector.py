import cv2
import mediapipe as mp
import math
from collections import deque, Counter


class CameraActivityDetector:

    def __init__(self):

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ====================================================
        # HISTORY
        # ====================================================

        self.hip_history = deque(maxlen=30)
        self.knee_history = deque(maxlen=30)
        self.ankle_history = deque(maxlen=30)

        self.prediction_history = deque(maxlen=8)


    # ========================================================
    # ANGLE
    # ========================================================

    def calculate_angle(self, a, b, c):

        angle = math.degrees(
            math.atan2(c.y - b.y, c.x - b.x)
            -
            math.atan2(a.y - b.y, a.x - b.x)
        )

        angle = abs(angle)

        if angle > 180:
            angle = 360 - angle

        return angle


    # ========================================================
    # SPEED
    # ========================================================

    def calculate_speed(self, history):

        if len(history) < 2:
            return 0

        total = 0

        for i in range(1, len(history)):

            previous = history[i - 1]
            current = history[i]

            total += math.sqrt(
                (current[0] - previous[0]) ** 2
                +
                (current[1] - previous[1]) ** 2
            )

        return total


    # ========================================================
    # VERTICAL MOVEMENT
    # ========================================================

    def calculate_vertical_movement(self, history):

        if len(history) < 2:
            return 0

        total = 0

        for i in range(1, len(history)):

            total += abs(
                history[i] - history[i - 1]
            )

        return total


    # ========================================================
    # ACTIVITY DETECTION
    # ========================================================

    def detect_activity(self, landmarks):

        mp_pose = self.mp_pose

        # ----------------------------------------------------
        # LANDMARKS
        # ----------------------------------------------------

        left_shoulder = landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        right_shoulder = landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        left_hip = landmarks[
            mp_pose.PoseLandmark.LEFT_HIP
        ]

        right_hip = landmarks[
            mp_pose.PoseLandmark.RIGHT_HIP
        ]

        left_knee = landmarks[
            mp_pose.PoseLandmark.LEFT_KNEE
        ]

        right_knee = landmarks[
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]

        left_ankle = landmarks[
            mp_pose.PoseLandmark.LEFT_ANKLE
        ]

        right_ankle = landmarks[
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        # ----------------------------------------------------
        # HIP CENTER
        # ----------------------------------------------------

        hip_x = (
            left_hip.x +
            right_hip.x
        ) / 2

        hip_y = (
            left_hip.y +
            right_hip.y
        ) / 2

        # ----------------------------------------------------
        # ANKLE CENTER
        # ----------------------------------------------------

        ankle_y = (
            left_ankle.y +
            right_ankle.y
        ) / 2

        # ----------------------------------------------------
        # KNEE ANGLES
        # ----------------------------------------------------

        left_angle = self.calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_angle = self.calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        average_knee_angle = (
            left_angle +
            right_angle
        ) / 2

        # ----------------------------------------------------
        # STORE HISTORY
        # ----------------------------------------------------

        self.hip_history.append(
            (hip_x, hip_y)
        )

        self.ankle_history.append(
            ankle_y
        )

        self.knee_history.append(
            (left_angle, right_angle)
        )

        # ----------------------------------------------------
        # WAIT FOR HISTORY
        # ----------------------------------------------------

        if len(self.hip_history) < 15:

            return "STANDING", 0.0

        # ----------------------------------------------------
        # MOVEMENT
        # ----------------------------------------------------

        hip_movement = self.calculate_speed(
            self.hip_history
        )

        ankle_movement = (
            self.calculate_vertical_movement(
                self.ankle_history
            )
        )

        # ----------------------------------------------------
        # KNEE MOVEMENT
        # ----------------------------------------------------

        knee_movement = 0

        for i in range(
            1,
            len(self.knee_history)
        ):

            previous = self.knee_history[i - 1]
            current = self.knee_history[i]

            knee_movement += (
                abs(
                    current[0] -
                    previous[0]
                )
                +
                abs(
                    current[1] -
                    previous[1]
                )
            )

        # ----------------------------------------------------
        # MOVEMENT SCORE
        # ----------------------------------------------------

        movement_score = (
            hip_movement * 1.5
            +
            ankle_movement * 2.0
            +
            knee_movement / 180
        )

        # ----------------------------------------------------
        # SITTING
        # ----------------------------------------------------

        if (
            average_knee_angle < 125
            and
            hip_movement < 0.35
        ):

            return "SITTING", movement_score

        # ----------------------------------------------------
        # RUNNING
        # ----------------------------------------------------

        if (
            movement_score > 1.70
            and
            knee_movement > 70
        ):

            return "RUNNING", movement_score

        # ----------------------------------------------------
        # WALKING
        # ----------------------------------------------------

        if (
            movement_score > 0.70
            and
            knee_movement > 20
        ):

            return "WALKING", movement_score

        # ----------------------------------------------------
        # STANDING
        # ----------------------------------------------------

        if (
            average_knee_angle > 140
            and
            hip_movement < 0.60
        ):

            return "STANDING", movement_score

        return "STANDING", movement_score


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Pose detection
        results = self.pose.process(
            rgb_frame
        )

        activity = "NO PERSON"
        movement_score = 0.0


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if results.pose_landmarks:

            landmarks = (
                results.pose_landmarks.landmark
            )

            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Detect activity
            activity, movement_score = (
                self.detect_activity(
                    landmarks
                )
            )

            # ----------------------------------------------
            # SMOOTH PREDICTION
            # ----------------------------------------------

            self.prediction_history.append(
                activity
            )

            if len(
                self.prediction_history
            ) > 8:

                self.prediction_history.popleft()

            activity = Counter(
                self.prediction_history
            ).most_common(1)[0][0]


        else:

            self.reset()


        return (
            frame,
            activity,
            movement_score
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.hip_history.clear()
        self.knee_history.clear()
        self.ankle_history.clear()
        self.prediction_history.clear()


    # ========================================================
    # RELEASE
    # ========================================================

    def close(self):

        self.pose.close()