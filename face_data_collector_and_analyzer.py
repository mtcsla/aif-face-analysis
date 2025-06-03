import cv2
from deepface import DeepFace
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
output_file = "collected_face_data.csv"
max_samples = 30
sample_count = 0

# === INIT CSV ===
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["age", "gender", "emotion"])

# === Start webcam and collect data ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🟢 Starting data collection...")

while sample_count < max_samples:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        try:
            results = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            age = results[0]['age']
            gender = results[0]['dominant_gender']
            emotion = results[0]['dominant_emotion']

            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([age, gender, emotion])
                sample_count += 1
                print(f"[{sample_count}/{max_samples}] Age: {age}, Gender: {gender}, Emotion: {emotion}")

        except Exception as e:
            print("⚠️ DeepFace error:", e)

        break

    cv2.putText(frame, f"Samples: {sample_count}/{max_samples}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Collecting...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Collection done. Analyzing...")

# === ANALYSIS ===
df = pd.read_csv(output_file)
gender_counts = df['gender'].value_counts()
emotion_counts = df['emotion'].value_counts()
average_age = df['age'].mean()

# Display benchmark-style output
print("\n===== BENCHMARK RESULTS =====")
print("Gender Distribution:")
print(gender_counts.to_string())
print("\nEmotion Distribution:")
print(emotion_counts.to_string())
print(f"\nAverage Age: {average_age:.2f}")

# Create and save plots
plt.figure(figsize=(6, 4))
gender_counts.plot(kind='bar', color='skyblue')
plt.title("Gender Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("gender_benchmark.png")
plt.show()

plt.figure(figsize=(8, 4))
emotion_counts.plot(kind='bar', color='salmon')
plt.title("Emotion Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("emotion_benchmark.png")
plt.show()

print("📊 Visualizations saved as PNG files.")
