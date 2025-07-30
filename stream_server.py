from flask import Flask, Response
import cv2
import os
import time
import constants as cons

app = Flask(__name__)

def generate_frames():
    last_frame = None
    while True:
        if os.path.exists(cons.IMAGE_PATH):
            frame = cv2.imread(cons.IMAGE_PATH)

            if frame is not None:
                if last_frame is None or not (frame == last_frame).all():
                    ret, buffer = cv2.imencode('.jpg', frame)
                    last_frame = frame.copy()
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
