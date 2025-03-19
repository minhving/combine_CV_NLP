import cv2
import mediapipe as mp
import math
import numpy as np
import pytesseract

# Initialize MediaPipe Hands solution and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)
draw_points = []
# Threshold for maximum allowed distance between consecutive points (in pixels)
THRESHOLD = 50

# Flag to prevent multiple recognitions per drawing session
recognition_done = False
recognized_text = ""

def fingers_folded(hand_landmarks):
    """
    Checks if the index, middle, ring, and pinky fingers are folded.
    For each finger, the tip should be below its PIP joint.
    """
    lm = hand_landmarks.landmark
    folded = True
    # Index: tip (8) vs. pip (6)
    if lm[8].y < lm[6].y:
        folded = False
    # Middle: tip (12) vs. pip (10)
    if lm[12].y < lm[10].y:
        folded = False
    # Ring: tip (16) vs. pip (14)
    if lm[16].y < lm[14].y:
        folded = False
    # Pinky: tip (20) vs. pip (18)
    if lm[20].y < lm[18].y:
        folded = False
    return folded

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If a hand is detected, process landmarks
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks on frame (optional)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get index finger tip and its PIP joint landmarks
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        # Only record point if the index finger is raised (tip above PIP)
        if index_tip.y < index_pip.y:
            draw_points.append((x, y))
            # Mark the current point (optional)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)
            recognition_done = False  # Reset recognition if drawing resumes

        # If all fingers are folded and we've collected enough points, trigger OCR once
        if fingers_folded(hand_landmarks) and not recognition_done and len(draw_points) > 10:
            # Create a blank canvas (black background)
            canvas = np.zeros((h, w), dtype=np.uint8)
            # Draw the trail on the canvas: white lines (value 255)
            for i in range(1, len(draw_points)):
                pt1 = draw_points[i - 1]
                pt2 = draw_points[i]
                if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < THRESHOLD:
                    cv2.line(canvas, pt1, pt2, 255, 3)
            # Optionally, dilate to thicken the stroke for better OCR
            kernel = np.ones((3, 3), np.uint8)
            canvas = cv2.dilate(canvas, kernel, iterations=1)
            # Invert the canvas so that text appears dark on a light background
            canvas_inv = cv2.bitwise_not(canvas)
            # Use pytesseract to extract the text
            recognized_text = pytesseract.image_to_string(canvas_inv, config='--psm 7').strip()
            print("Recognized:", recognized_text)
            recognition_done = True

    # Draw the trail by connecting consecutive points if they are close enough
    for i in range(1, len(draw_points)):
        pt1 = draw_points[i - 1]
        pt2 = draw_points[i]
        if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < THRESHOLD:
            cv2.line(frame, pt1, pt2, (255, 0, 0), 3)

    # Display recognized text on the frame
    cv2.putText(frame, recognized_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Hand Drawing Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear drawing and recognized text
        draw_points = []
        recognized_text = ""
        recognition_done = False
    elif key == ord('q'):  # Quit the application
        break

cap.release()
cv2.destroyAllWindows()
