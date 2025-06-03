import cv2

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

def get_emotion_color(emotion):
    """Return BGR color tuple for a given emotion."""
    emotion_lower = str(emotion).lower()
    if emotion_lower == "happy":
        return (0, 255, 0)      # Green
    elif emotion_lower == "neutral":
        return (128, 128, 128)  # Gray
    elif emotion_lower == "angry":
        return (0, 0, 255)      # Red (OpenCV uses BGR)
    elif emotion_lower == "sad":
        return (255, 0, 0)      # Blue
    else:
        return (0, 255, 255)    # Yellow for other emotions
