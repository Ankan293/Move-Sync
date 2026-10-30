import cv2
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from camera_detector import CameraActivityDetector


# ============================================================
# MOVE SYNC CAMERA SERVER
# ============================================================

HOST = "127.0.0.1"
PORT = 8765


detector = CameraActivityDetector()

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")


latest_frame = None

latest_status = {
    "activity": "NO PERSON",
    "movement": 0.0,
    "running": True
}

lock = threading.Lock()


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_frame
    global latest_status

    while True:

        success, frame = camera.read()

        if not success:
            continue

        frame, activity, movement = detector.process_frame(frame)

        # Add activity text to camera frame
        cv2.rectangle(
            frame,
            (10, 10),
            (630, 90),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"ACTIVITY: {activity}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Movement: {movement:.2f}",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # Encode frame as JPEG
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = encoded.tobytes()

        with lock:

            latest_frame = frame_bytes

            latest_status = {
                "activity": activity,
                "movement": float(movement),
                "running": True
            }


# ============================================================
# HTTP SERVER
# ============================================================

class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if self.path == "/status":

            with lock:
                data = json.dumps(
                    latest_status
                ).encode("utf-8")

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.send_header(
                "Content-Length",
                str(len(data))
            )

            self.end_headers()

            self.wfile.write(data)

            return


        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        if self.path == "/frame":

            with lock:
                frame = latest_frame

            if frame is None:

                self.send_response(503)
                self.end_headers()

                return

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "image/jpeg"
            )

            self.send_header(
                "Content-Length",
                str(len(frame))
            )

            self.end_headers()

            self.wfile.write(frame)

            return


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if self.path == "/":

            message = (
                "MOVE SYNC Camera Server is running.\n"
                "/status -> activity status\n"
                "/frame -> latest camera frame\n"
            ).encode("utf-8")

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/plain"
            )

            self.send_header(
                "Content-Length",
                str(len(message))
            )

            self.end_headers()

            self.wfile.write(message)

            return


        self.send_response(404)
        self.end_headers()


    def log_message(self, format, *args):
        return


# ============================================================
# START
# ============================================================

thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

thread.start()


server = ThreadingHTTPServer(
    (HOST, PORT),
    RequestHandler
)

print()
print("=" * 60)
print("MOVE SYNC CAMERA SERVER")
print("=" * 60)
print()
print("Camera: ACTIVE")
print("Server: http://127.0.0.1:8765")
print()
print("Keep this terminal running.")
print("Press CTRL+C to stop.")
print()


try:

    server.serve_forever()

except KeyboardInterrupt:

    print("\nStopping camera server...")

finally:

    camera.release()
    detector.close()
    server.server_close()