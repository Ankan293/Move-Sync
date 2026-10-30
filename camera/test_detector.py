import cv2

from camera_detector import CameraActivityDetector


detector = CameraActivityDetector()

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print("MOVE SYNC CAMERA DETECTOR")
print("Press Q to quit.")


while True:

    ret, frame = camera.read()

    if not ret:
        break


    frame, activity, movement = (
        detector.process_frame(frame)
    )


    # Display activity
    cv2.rectangle(
        frame,
        (15, 55),
        (500, 155),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"ACTIVITY: {activity}",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Movement: {movement:.2f}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "MOVE SYNC Camera Detector",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

detector.close()

cv2.destroyAllWindows()