import time
from deepface import DeepFace
import cv2
import numpy as np

def process_frame_for_analysis(frame, target_width=320):
    """Resize frame for analysis, preserving aspect ratio. Returns (small_frame, scale_x, scale_y)."""
    orig_h, orig_w = frame.shape[:2]
    target_w = target_width
    target_h = int(orig_h * target_w / orig_w)
    small_frame = cv2.resize(frame, (target_w, target_h))
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    return small_frame, scale_x, scale_y

def analyze_emotion_every_n_frames(small_frame, frame_counter, emotion_result_cache, emotion_cache, region_cache, n=5):
    """Analyze emotion every n frames, reuse cached result otherwise."""
    if frame_counter % n == 0 or emotion_result_cache is None:
        emotion_result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')[0]
        emotion_result_cache = emotion_result
        emotion_cache = emotion_result['dominant_emotion']
        region_cache = emotion_result.get('region')
    return emotion_result_cache, emotion_cache, region_cache

def analyze_age_gender_once_per_second(small_frame, last_age, last_gender, last_age_gender_time):
    """Analyze age/gender once per second, reuse last result otherwise."""
    now = time.time()
    if now - last_age_gender_time > 1:
        age_gender_result = DeepFace.analyze(small_frame, actions=['age', 'gender'], enforce_detection=False, detector_backend='opencv')[0]
        last_age = age_gender_result.get('age', None)
        last_gender = age_gender_result.get('dominant_gender', None)
        last_age_gender_time = now
    return last_age, last_gender, last_age_gender_time

def compose_label(last_gender, last_age, emotion):
    """Compose label for display."""
    if last_gender is not None and last_age is not None:
        return f"{last_gender}, {last_age}, {emotion}"
    else:
        return f"{emotion}"

def analyze_full_image(img):
    """Analyze uploaded image for age, gender, emotion. Returns (result_text, img_rgb)."""
    result_text = "Error: Unable to analyze"
    try:
        analysis = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        if analysis:
            result = analysis[0]
            result_text = f"Age: {result['age']}\nGender: {result['dominant_gender']}\nEmotion: {result['dominant_emotion']}"
    except Exception as e:
        result_text = f"Error: {str(e)}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return result_text, img_rgb
