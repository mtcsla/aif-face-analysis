import tkinter as tk
from tkinter import filedialog
from typing import Optional

import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

cap: Optional[cv2.VideoCapture] = None
video_label: Optional[tk.Label] = None
running = False

def analyze_uploaded_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    result_text = "Error: Unable to analyze"
    try:
        analysis = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        if analysis:
            result = analysis[0]
            result_text = f"Age: {result['age']}\nGender: {result['dominant_gender']}\nEmotion: {result['dominant_emotion']}"
    except Exception as e:
        result_text = f"Error: {str(e)}"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)

    win = tk.Toplevel(root)
    win.title("Analysis Result")
    panel = tk.Label(win, image=img_tk)
    panel.pack()
    tk.Label(win, text=result_text, font=("Helvetica", 12)).pack(pady=10)

import time

last_age = None
last_gender = None
last_age_gender_time = 0

# For frame skipping and result caching
frame_counter = 0
emotion_result_cache = None
emotion_cache = None
region_cache = None

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2

    if thickness < 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    # Draw straight lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw arcs
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

def update_camera():
    global cap, running, last_age, last_gender, last_age_gender_time
    global frame_counter, emotion_result_cache, emotion_cache, region_cache

    if not running:
        return
    if cap:
        ret, frame = cap.read()
    else:
        return

    frame = cv2.flip(frame, 1)

    # Resize for faster analysis, preserving aspect ratio
    orig_h, orig_w = frame.shape[:2]
    target_w, target_h = 320, int(orig_h * 320 / orig_w)
    small_frame = cv2.resize(frame, (target_w, target_h))
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h

    # Analyze emotion every 5th frame, reuse result otherwise
    if frame_counter % 5 == 0 or emotion_result_cache is None:
        emotion_result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')[0]
        emotion_result_cache = emotion_result
        emotion_cache = emotion_result['dominant_emotion']
        region_cache = emotion_result.get('region')
    emotion = emotion_cache
    region = region_cache

    # Analyze age/gender once per second (on small frame)
    now = time.time()
    if now - last_age_gender_time > 1:
        age_gender_result = DeepFace.analyze(small_frame, actions=['age', 'gender'], enforce_detection=False, detector_backend='opencv')[0]
        last_age = age_gender_result.get('age', None)
        last_gender = age_gender_result.get('dominant_gender', None)
        last_age_gender_time = now

    # Compose label
    if last_gender is not None and last_age is not None:
        label = f"{last_gender}, {last_age}, {emotion}"
    else:
        label = f"{emotion}"

    # Draw rectangle and label above face if region info is available
    region = region_cache
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

        # Set color based on emotion
        emotion_lower = emotion.lower()
        if emotion_lower == "happy":
            box_color = (0, 255, 0)      # Green
        elif emotion_lower == "neutral":
            box_color = (128, 128, 128)  # Gray
        elif emotion_lower == "angry":
            box_color = (0, 0, 255)      # Red (OpenCV uses BGR)
        elif emotion_lower == "sad":
            box_color = (255, 0, 0)      # Blue
        else:
            box_color = (0, 255, 255)    # Yellow for other emotions

        draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, 2, radius=15)
        label_y = max(y - 10, 20)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    else:
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    if video_label:
        video_label.config(image=imgtk)
        video_label.image = imgtk

    frame_counter += 1
    root.after(40, update_camera)

def analyze_live_camera():
    global cap, video_label, running
    running = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return
    win = tk.Toplevel(root)
    win.title("Live Detection")
    video_label = tk.Label(win)
    video_label.pack()

    def stop():
        global running
        running = False
        if cap:
            cap.release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", stop)
    update_camera()

root = tk.Tk()
root.title("Face Recognition UI")
root.geometry("300x200")

tk.Button(root, text="Analyze Uploaded Photo", command=analyze_uploaded_image,
          font=("Helvetica", 12), width=25).pack(pady=20)
tk.Button(root, text="Start Live Camera", command=analyze_live_camera,
          font=("Helvetica", 12), width=25).pack(pady=10)

root.mainloop()
