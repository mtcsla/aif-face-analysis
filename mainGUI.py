import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image, ImageTk

face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def analyze_uploaded_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    emotion, score = "No face", 0

    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if analysis:
            emotion = analysis[0]['dominant_emotion']
            score = analysis[0]['emotion'][emotion] / 100.0
    except:
        emotion = "Error"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)

    result_text = f"Emotion: {emotion} ({int(score * 100)}%)"
    result_window = tk.Toplevel(root)
    result_window.title("Emotion Analysis")
    panel = tk.Label(result_window, image=img_tk)
    panel.image = img_tk
    panel.pack()
    label = tk.Label(result_window, text=result_text, font=("Helvetica", 14))
    label.pack(pady=10)

def analyze_live_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as fd:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)
                    face = frame[y:y2, x:x2]
                    emotion, score = "No detection", 0

                    try:
                        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                        if result:
                            emotion = result[0]['dominant_emotion']
                            score = result[0]['emotion'][emotion] / 100.0
                    except:
                        pass

                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion} ({int(score * 100)}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Live Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Face Emotion App")
root.geometry("300x200")

tk.Button(root, text="Analyze Uploaded Photo", command=analyze_uploaded_image,
          font=("Helvetica", 12), width=25).pack(pady=20)
tk.Button(root, text="Start Live Camera", command=analyze_live_camera,
          font=("Helvetica", 12), width=25).pack(pady=10)

root.mainloop()
