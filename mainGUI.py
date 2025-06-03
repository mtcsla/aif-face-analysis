import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

cap = None
video_label = None
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
    panel.image = img_tk
    panel.pack()
    tk.Label(win, text=result_text, font=("Helvetica", 12)).pack(pady=10)

def update_camera():
    global cap, running
    if not running:
        return
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
        label = f"{result['dominant_gender']}, {result['age']}, {result['dominant_emotion']}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, "Analysis Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(200, update_camera)

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
