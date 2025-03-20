from imports import *
import cv2
import mediapipe as mp
import math
import numpy as np
import pytesseract
import time
import threading

model = RagOpenAI()
model.initialize_openai()

# Initialize MediaPipe Hands solution and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)
draw_points = []
THRESHOLD = 50  # Maximum allowed distance between consecutive points (in pixels)

# Flags and counters for recognition and asynchronous model processing
recognition_done = False
recognized_text = ""
old_recognized_text = ""
result = ""   # This holds the final result from the model.
count = 0
processing = False  # Indicates if the model is currently processing
drawing = True

count1 = 0
count2 = 0
count3 = 0
def fingers_folded(hand_landmarks):
    """
    Checks if the index, middle, ring, and pinky fingers are folded.
    Each finger is considered folded if its tip is below its PIP joint.
    """
    lm = hand_landmarks.landmark
    folded = True
    if lm[8].y < lm[6].y:
        folded = False
    if lm[12].y < lm[10].y:
        folded = False
    if lm[16].y < lm[14].y:
        folded = False
    if lm[20].y < lm[18].y:
        folded = False
    return folded

def process_model(text):
    """
    Calls the model's reply function and updates the global result.
    Runs in a separate thread.
    """
    global result, processing, draw_points
    result = model.reply_from_chat_bot(text)
    processing = False
    # Optionally, clear drawing points after processing.
    draw_points = []

# Set up full-screen window
window_name = "Hand Drawing Recognition"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get index finger tip and its PIP joint landmarks.
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        # Record drawing points if the index finger is raised (tip above PIP).
        if index_tip.y < index_pip.y and drawing:
            draw_points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)
            recognition_done = False  # Reset recognition if drawing resumes.

        # When all fingers are folded and enough points have been collected, trigger OCR.
        if fingers_folded(hand_landmarks) and not recognition_done and len(draw_points) > 10:
            canvas = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, len(draw_points)):
                pt1 = draw_points[i - 1]
                pt2 = draw_points[i]
                if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < THRESHOLD:
                    cv2.line(canvas, pt1, pt2, 255, 3)
            kernel = np.ones((3, 3), np.uint8)
            canvas = cv2.dilate(canvas, kernel, iterations=1)
            canvas_inv = cv2.bitwise_not(canvas)
            recognized_text = pytesseract.image_to_string(canvas_inv, config='--psm 7').strip()
            print("Recognized:", recognized_text)
            recognition_done = True

        # When fingers remain folded and recognition is done, check if we should trigger the model.
        if fingers_folded(hand_landmarks) and recognition_done:
            if recognized_text == old_recognized_text:
                count += 1
                if count > 50 and not processing:
                    processing = True
                    threading.Thread(target=process_model, args=(recognized_text,)).start()
                    print(result)
                    drawing = False
                    count = 0
            else:
                old_recognized_text = recognized_text
                count = 0

    # Draw the user's trail on the frame.
    for i in range(1, len(draw_points)):
        pt1 = draw_points[i - 1]
        pt2 = draw_points[i]
        if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < THRESHOLD:
            cv2.line(frame, pt1, pt2, (255, 0, 0), 3)

    # Display the result persistently if available.
    if result:
        title_index = []
        video_id = []
        for i in range(len(result)):
            if i + 6 < len(result):
                if result[i:i + 6] == "Title:":
                    title_index.append(i)
            if i + 9 < len(result):
                if result[i:i + 9] == "Video ID:":
                    video_id.append(i)
        title = []
        ids = []
        for i in range(len(title_index)):
            if i + 1 < len(title_index):
                title.append(result[title_index[i] + 6:video_id[i] - 1].strip())
                ids.append(result[video_id[i] + 9:title_index[i + 1] - 3].strip())
            else:
                title.append(result[title_index[i] + 6:video_id[i] - 1].strip())
                ids.append(result[video_id[i] + 9:].strip())
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                if lm[12].y < lm[10].y and lm[16].y < lm[14].y and lm[8].y < lm[6].y:
                    count1 += 1
                    count2 = 0
                    count3 = 0
                    cv2.putText(frame, title[0], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[1], (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[2], (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    if count1 == 20:
                        url = "https://www.youtube.com/watch?v=" + ids[2]
                        webbrowser.open(url)
                        break
                elif lm[12].y < lm[10].y and lm[8].y < lm[6].y:
                    count2 += 1
                    count1 = 0
                    count3 = 0
                    cv2.putText(frame, title[0], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[1], (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[2], (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    if count2 == 20:
                        url = "https://www.youtube.com/watch?v=" + ids[1]
                        webbrowser.open(url)
                        break
                elif lm[8].y < lm[6].y:
                    count3 += 1
                    count1 = 0
                    count2 = 0
                    cv2.putText(frame, title[0], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[1], (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[2], (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    if count3 == 20:
                        url = "https://www.youtube.com/watch?v=" + ids[0]
                        webbrowser.open(url)
                        break
                else:
                    cv2.putText(frame, title[0], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[1], (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, title[2], (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)


    elif processing:
        cv2.putText(frame, "Processing...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, recognized_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear drawing and reset text/flags.
        draw_points = []
        recognized_text = ""
        recognition_done = False
        result = ""
        count = 0
    elif key == ord('q'):  # Quit the application.
        break

    time.sleep(0.1)  # Slow down loop iterations slightly.

cap.release()
cv2.destroyAllWindows()
