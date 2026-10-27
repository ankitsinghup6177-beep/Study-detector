import base64
import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize AI models once on server startup
face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO("yolov8n.pt")
classNames = phone_detector.names

# Landmark Indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

# Thresholds
SLEEP_THRESHOLD_FRAMES = 5    # Reduced for HTTP frame rate
COVER_THRESHOLD_FRAMES = 6

# Per-session state tracking (simple in-memory storage)
user_states = {}

def decode_base64_image(image_data):
    """Converts base64 string from browser into OpenCV BGR image"""
    try:
        header, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'clear'})

    session_id = data.get('session_id', 'default_user')
    img = decode_base64_image(data['image'])

    if img is None:
        return jsonify({'status': 'clear'})

    # Initialize state tracking for new sessions
    if session_id not in user_states:
        user_states[session_id] = {'closed_frames': 0, 'covered_frames': 0}

    state = user_states[session_id]
    is_sleepy = False
    is_face_covered = False
    phone_detected = False

    # 1. Face & Sleep Detection
    img, faces = face_detector.findFaceMesh(img, draw=False)

    if faces:
        state['covered_frames'] = 0
        face = faces[0]

        eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
        face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])

        ratio = (eye_dist / face_dist) * 100 if face_dist > 0 else 100

        if ratio < 11.0:
            state['closed_frames'] += 1
        else:
            state['closed_frames'] = 0

        if state['closed_frames'] >= SLEEP_THRESHOLD_FRAMES:
            is_sleepy = True
    else:
        state['closed_frames'] = 0
        state['covered_frames'] += 1
        if state['covered_frames'] >= COVER_THRESHOLD_FRAMES:
            is_face_covered = True

    # 2. Phone Detection
    results = phone_detector.predict(img, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if classNames[cls_id] == "cell phone" and conf > 0.5:
                phone_detected = True
                break

    # 3. Determine Final Status
    if is_face_covered:
        status = "face_covered"
    elif is_sleepy:
        status = "sleepy"
    elif phone_detected:
        status = "phone"
    else:
        status = "clear"

    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)