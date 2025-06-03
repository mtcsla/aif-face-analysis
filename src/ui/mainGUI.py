import tkinter as tk
from tkinter import filedialog
from typing import Optional
import cv2
from PIL import Image, ImageTk

from logic.face_analysis import (
    analyze_emotion_every_n_frames,
    analyze_age_gender_once_per_second,
    process_frame_for_analysis,
    compose_label
)
from utils.draw import draw_rounded_rectangle, get_emotion_color

cap: Optional[cv2.VideoCapture] = None
video_label: Optional[tk.Label] = None
running = False

# State for analysis
last_age = None
last_gender = None
last_age_gender_time = 0
frame_counter = 0
emotion_result_cache = None
emotion_cache = None
region_cache = None

def analyze_uploaded_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    from logic.face_analysis import analyze_full_image
    result_text, img_rgb = analyze_full_image(img)
    img_pil = Image.fromarray(img_rgb).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)

    win = tk.Toplevel(root)
    win.title("Analysis Result")
    panel = tk.Label(win, image=img_tk)
    panel.__dict__["image"] = img_tk
    panel.pack()
    tk.Label(win, text=result_text, font=("Helvetica", 12)).pack(pady=10)

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
    orig_h, orig_w = frame.shape[:2]

    # Process frame for analysis (resize, scale factors)
    small_frame, scale_x, scale_y = process_frame_for_analysis(frame)

    # Analyze emotion every 5th frame, reuse result otherwise
    emotion_result_cache, emotion_cache, region_cache = analyze_emotion_every_n_frames(
        small_frame, frame_counter, emotion_result_cache, emotion_cache, region_cache
    )
    emotion = emotion_cache
    region = region_cache

    # Analyze age/gender once per second (on small frame)
    last_age, last_gender, last_age_gender_time = analyze_age_gender_once_per_second(
        small_frame, last_age, last_gender, last_age_gender_time
    )

    # Compose label
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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    if video_label:
        video_label.config(image=imgtk)
        video_label.__dict__["image"] = imgtk

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

def launch_ui():
    global root
    root = tk.Tk()
    root.title("Face Recognition UI")
    root.geometry("300x200")

    tk.Button(root, text="Analyze Uploaded Photo", command=analyze_uploaded_image,
              font=("Helvetica", 12), width=25).pack(pady=20)
    tk.Button(root, text="Start Live Camera", command=analyze_live_camera,
              font=("Helvetica", 12), width=25).pack(pady=10)

    root.mainloop()
