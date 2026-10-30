import cv2
import mediapipe as mp


# ============================================================
# MEDIAPIPE SETUP
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

print("MOVE SYNC Pose Detection Started")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Flip camera horizontally
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect pose
    results = pose.process(rgb_frame)


    # ========================================================
    # DRAW BODY LANDMARKS
    # ========================================================

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # ====================================================
        # GET LANDMARKS
        # ====================================================

        landmarks = results.pose_landmarks.landmark

        nose = landmarks[
            mp_pose.PoseLandmark.NOSE
        ]

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


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        cv2.putText(
            frame,
            "PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display some landmark coordinates
        cv2.putText(
            frame,
            f"Nose: ({nose.x:.2f}, {nose.y:.2f})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Left Hip: ({left_hip.x:.2f}, {left_hip.y:.2f})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "MOVE SYNC - Pose Detection",
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

print("Pose detection stopped.")