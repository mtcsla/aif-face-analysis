import tkinter as tk
from tkinter import filedialog
import cv2
from deepface import DeepFace
from PIL import Image, ImageTk
import numpy as np

# === Function: Analyze image from file ===
def analyze_uploaded_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Read and analyze image
    img = cv2.imread(file_path)
    try:
        results = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
        display_result(img, results)
    except Exception as e:
        print("Analysis failed:", e)

# === Function: Live camera analysis ===
def analyze_live_camera():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                results = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
                age = results['age']
                gender = results['dominant_gender']
                emotion = results['dominant_emotion']

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                pass

        cv2.imshow("Live Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Helper: Show analyzed result for image ===
def display_result(img, results):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300))

    result_text = f"Age: {results['age']}\nGender: {results['dominant_gender']}\nEmotion: {results['dominant_emotion']}"

    result_window = tk.Toplevel(root)
    result_window.title("Analysis Result")

    img_tk = ImageTk.PhotoImage(img_pil)
    panel = tk.Label(result_window, image=img_tk)
    panel.image = img_tk
    panel.pack()

    label = tk.Label(result_window, text=result_text, font=("Helvetica", 14))
    label.pack(pady=10)

# === GUI Setup ===
root = tk.Tk()
root.title("Face Analysis App")
root.geometry("300x200")

btn_upload = tk.Button(root, text="Analyze Uploaded Photo", command=analyze_uploaded_image, font=("Helvetica", 12), width=25)
btn_upload.pack(pady=20)

btn_live = tk.Button(root, text="Start Live Camera Analysis", command=analyze_live_camera, font=("Helvetica", 12), width=25)
btn_live.pack(pady=10)

root.mainloop()
