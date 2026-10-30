import cv2
import mediapipe as mp
import math
import time
from collections import deque, Counter


# ============================================================
# MOVE SYNC
# REAL-TIME ACTIVITY RECOGNITION
# ============================================================


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# ============================================================
# HISTORY
# ============================================================

position_history = deque(maxlen=30)
knee_history = deque(maxlen=30)
ankle_history = deque(maxlen=30)

prediction_history = deque(maxlen=8)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        -
        math.atan2(a.y - b.y, a.x - b.x)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def calculate_speed(history):

    if len(history) < 2:
        return 0

    total = 0

    for i in range(1, len(history)):

        previous = history[i - 1]
        current = history[i]

        distance = math.sqrt(
            (current[0] - previous[0]) ** 2 +
            (current[1] - previous[1]) ** 2
        )

        total += distance

    return total


def calculate_vertical_movement(history):

    if len(history) < 2:
        return 0

    total = 0

    for i in range(1, len(history)):

        total += abs(
            history[i] - history[i - 1]
        )

    return total


# ============================================================
# ACTIVITY DETECTION
# ============================================================

def detect_activity(landmarks):

    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CENTERS
    # --------------------------------------------------------

    hip_x = (
        left_hip.x +
        right_hip.x
    ) / 2

    hip_y = (
        left_hip.y +
        right_hip.y
    ) / 2


    ankle_x = (
        left_ankle.x +
        right_ankle.x
    ) / 2

    ankle_y = (
        left_ankle.y +
        right_ankle.y
    ) / 2


    # --------------------------------------------------------
    # KNEE ANGLES
    # --------------------------------------------------------

    left_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    average_knee_angle = (
        left_angle +
        right_angle
    ) / 2


    # --------------------------------------------------------
    # TORSO
    # --------------------------------------------------------

    shoulder_x = (
        left_shoulder.x +
        right_shoulder.x
    ) / 2

    shoulder_y = (
        left_shoulder.y +
        right_shoulder.y
    ) / 2

    torso_dx = hip_x - shoulder_x
    torso_dy = hip_y - shoulder_y

    torso_angle = abs(
        math.degrees(
            math.atan2(
                torso_dx,
                torso_dy
            )
        )
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    position_history.append(
        (hip_x, hip_y)
    )

    ankle_history.append(
        ankle_y
    )

    knee_history.append(
        (
            left_angle,
            right_angle
        )
    )


    # --------------------------------------------------------
    # NOT ENOUGH HISTORY
    # --------------------------------------------------------

    if len(position_history) < 15:

        return "STANDING", 0


    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    hip_movement = calculate_speed(
        position_history
    )

    ankle_movement = calculate_vertical_movement(
        ankle_history
    )


    # --------------------------------------------------------
    # KNEE MOVEMENT
    # --------------------------------------------------------

    knee_movement = 0

    for i in range(
        1,
        len(knee_history)
    ):

        previous = knee_history[i - 1]
        current = knee_history[i]

        knee_movement += (
            abs(current[0] - previous[0]) +
            abs(current[1] - previous[1])
        )


    # --------------------------------------------------------
    # COMBINED MOVEMENT SCORE
    # --------------------------------------------------------

    movement_score = (
        hip_movement * 1.5 +
        ankle_movement * 2.0 +
        knee_movement / 180
    )


    # --------------------------------------------------------
    # SITTING
    # --------------------------------------------------------

    if (
        average_knee_angle < 125
        and
        hip_movement < 0.35
    ):

        return "SITTING", movement_score


    # --------------------------------------------------------
    # RUNNING
    #
    # Running normally creates significantly larger
    # leg movement than walking.
    # --------------------------------------------------------

    if (
        movement_score > 1.70
        and
        knee_movement > 70
    ):

        return "RUNNING", movement_score


    # --------------------------------------------------------
    # WALKING
    # --------------------------------------------------------

    if (
        movement_score > 0.7
        and
        knee_movement > 20
    ):

        return "WALKING", movement_score


    # --------------------------------------------------------
    # STANDING
    # --------------------------------------------------------

    if (
        average_knee_angle > 140
        and
        hip_movement < 0.6
    ):

        return "STANDING", movement_score


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "STANDING", movement_score


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 60)
print("MOVE SYNC - REAL-TIME ACTIVITY RECOGNITION")
print("=" * 60)
print()
print("Stand, sit, walk or run in front of the camera.")
print("Press Q to quit.")
print()


while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")
        break


    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    results = pose.process(
        rgb_frame
    )


    activity = "NO PERSON"
    movement_score = 0


    # ========================================================
    # PERSON DETECTED
    # ========================================================

    if results.pose_landmarks:

        landmarks = (
            results.pose_landmarks.landmark
        )


        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # Detect activity
        activity, movement_score = detect_activity(
            landmarks
        )


        # ----------------------------------------------------
        # SMOOTH PREDICTION
        # ----------------------------------------------------

        prediction_history.append(
            activity
        )

        if len(prediction_history) >= 4:

            activity = Counter(
                prediction_history
            ).most_common(1)[0][0]


        # ----------------------------------------------------
        # PERSON STATUS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "PERSON DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    else:

        position_history.clear()
        knee_history.clear()
        ankle_history.clear()
        prediction_history.clear()


    # ========================================================
    # ACTIVITY DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (15, 55),
        (560, 165),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "ACTIVITY",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        activity,
        (35, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 255),
        3
    )


    # ========================================================
    # MOVEMENT SCORE
    # ========================================================

    cv2.putText(
        frame,
        f"Movement: {movement_score:.2f}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "MOVE SYNC - Real-Time Activity Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

pose.close()

cv2.destroyAllWindows()

print()
print("MOVE SYNC stopped.")