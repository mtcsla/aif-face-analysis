import cv2
import mediapipe as mp
from fer import FER

emotion_detector = FER(mtcnn=False)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Extract face
                face_roi = frame[y:y+h, x:x+w]

                # Use FER for emotion
                emotion, score = None, 0
                try:
                    result = emotion_detector.top_emotion(face_roi)
                    if result:
                        emotion, score = result
                except Exception as e:
                    emotion = "Error"

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"Emotion: {emotion} ({int(score*100)}%)" if emotion else "No emotion"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        cv2.imshow("MediaPipe + FER (Emotion)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
