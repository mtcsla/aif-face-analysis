import cv2
import mediapipe as mp
from deepface import DeepFace

mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(0, int(bbox.xmin * iw))
                y = max(0, int(bbox.ymin * ih))
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x2 = min(iw, x + w)
                y2 = min(ih, y + h)

                face_roi = frame[y:y2, x:x2]
                emotion, score = "None", 0

                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if result:
                        emotion = result[0]['dominant_emotion']
                        score = result[0]['emotion'][emotion] / 100.0
                except:
                    emotion = "Error"

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                label = f"Emotion: {emotion} ({int(score * 100)}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        cv2.imshow("MediaPipe + DeepFace (Emotion)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
