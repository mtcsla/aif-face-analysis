from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import cv2
import numpy as np
import secrets
import os
from dotenv import load_dotenv

from .logic.face_analysis import (
    process_frame_for_analysis,
    analyze_emotion_every_n_frames,
    analyze_age_gender_once_per_second,
    compose_label
)
from .utils.draw import draw_rounded_rectangle, get_emotion_color

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
security = HTTPBasic()

APP_USERNAME = os.getenv("APP_USERNAME", "user")
APP_PASSWORD = os.getenv("APP_PASSWORD", "password")

def check_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, APP_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, APP_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

@app.get("/", response_class=HTMLResponse)
async def index(credentials: HTTPBasicCredentials = Depends(check_basic_auth)):
    return FileResponse("resources/index.html", media_type="text/html")

def analyze_and_annotate_frame(frame, state):
    orig_h, orig_w = frame.shape[:2]
    small_frame, scale_x, scale_y = process_frame_for_analysis(frame)

    # Analyze emotion every 5th frame, reuse result otherwise
    state['emotion_result_cache'], state['emotion_cache'], state['region_cache'] = analyze_emotion_every_n_frames(
        small_frame, state['frame_counter'], state['emotion_result_cache'], state['emotion_cache'], state['region_cache']
    )
    emotion = state['emotion_cache']
    region = state['region_cache']

    # Analyze age/gender once per second (on small frame)
    state['last_age'], state['last_gender'], state['last_age_gender_time'] = analyze_age_gender_once_per_second(
        small_frame, state['last_age'], state['last_gender'], state['last_age_gender_time']
    )

    # Compose label
    label = compose_label(state['last_gender'], state['last_age'], emotion)

    # Draw rectangle and label above face if region info is available
    if region:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        # Scale region to original frame size
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        # Ensure rectangle stays within frame bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))

        box_color = get_emotion_color(emotion)
        draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, 2, radius=15)
        label_y = max(y - 10, 20)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    else:
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    state['frame_counter'] += 1
    return frame

def analyze_and_annotate_static_image(frame):
    # Use the same logic as analyze_and_annotate_frame, but with a fresh state and always analyze
    orig_h, orig_w = frame.shape[:2]
    small_frame, scale_x, scale_y = process_frame_for_analysis(frame)
    # Always analyze emotion, age, gender
    from .logic.face_analysis import analyze_emotion_every_n_frames, analyze_age_gender_once_per_second, compose_label
    emotion_result = None
    emotion_cache = None
    region_cache = None
    emotion_result, emotion_cache, region_cache = analyze_emotion_every_n_frames(
        small_frame, 0, None, None, None, n=1
    )
    emotion = emotion_cache
    region = region_cache
    last_age, last_gender, _ = analyze_age_gender_once_per_second(
        small_frame, None, None, 0
    )
    label = compose_label(last_gender, last_age, emotion)
    # Draw rectangle and label above face if region info is available
    if region:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        # Scale region to original frame size
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        # Ensure rectangle stays within frame bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))
        box_color = get_emotion_color(emotion)
        draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, 2, radius=15)
        label_y = max(y - 10, 20)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    else:
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

@app.websocket("/ws-analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # State for analysis
    state = {
        'last_age': None,
        'last_gender': None,
        'last_age_gender_time': 0,
        'frame_counter': 0,
        'emotion_result_cache': None,
        'emotion_cache': None,
        'region_cache': None,
    }
    try:
        while True:
            data = await websocket.receive_bytes()
            # Decode JPEG to frame
            npimg = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)  # Mirror the image horizontally
            frame = cv2.resize(frame, (640, 480))
            annotated = analyze_and_annotate_frame(frame, state)
            _, jpeg = cv2.imencode('.jpg', annotated)
            await websocket.send_bytes(jpeg.tobytes())
    except WebSocketDisconnect:
        pass

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
   # Read image file
   contents = await file.read()
   npimg = np.frombuffer(contents, np.uint8)
   frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
   if frame is None:
       raise HTTPException(status_code=400, detail="Invalid image file")
   frame = cv2.resize(frame, (640, 480))
   annotated = analyze_and_annotate_static_image(frame)
   _, jpeg = cv2.imencode('.jpg', annotated)
   return Response(content=jpeg.tobytes(), media_type="image/jpeg")
