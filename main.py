from imports import *
import cv2
import mediapipe as mp
import math
import numpy as np
import pytesseract
import time
import threading
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# #load predict price
model_predict = EnsembleAgent()
model_predict.initialize()
detected_object = None


# Initialize MediaPipe Hands solution and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
window_name = "Object detection"

#initilize function
count1 = 0
count2 = 0
count3 = 0

#detection function
count4 = 0

#list object
count5 = 0
count6 = 0
count7 = 0
choice = -1

initialize = True
prediction = False
processing = False
result = None
with open("data.json", "r") as f:
    data = json.load(f)
def process_model(text):
    """
    Calls the model's reply function and updates the global result.
    Runs in a separate thread.
    """
    global result, processing
    result = model_predict.price(text)
    print(result)
    processing = False
    # Optionally, clear drawing points after processing.
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


    if initialize == True:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                if lm[16].y < lm[14].y and lm[12].y < lm[10].y and lm[8].y < lm[6].y:
                    count3 += 1
                    print(count3)
                    count2 = 0
                    count1 = 0
                    cv2.putText(frame, "Welcome to the program", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "1.Detect the object", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "2. Prediction function", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "3. List of detected object", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    if count3 == 20:
                        initialize = False
                elif lm[12].y < lm[10].y and lm[8].y< lm[6].y:
                    count2 += 1
                    count1 = 0
                    count3 = 0
                    cv2.putText(frame, "Welcome to the program", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "1.Detect the object", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "2. Prediction function", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "3. List of detected object", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    if count2 == 20:
                        initialize = False
                elif lm[8].y < lm[6].y:
                    count1 += 1
                    count2 = 0
                    count3 = 0
                    cv2.putText(frame, "Welcome to the program", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "1.Detect the object", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "2. Prediction function", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "3. List of detected object", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    if count1 == 20:
                        initialize = False
                        prediction = True
        else:
            cv2.putText(frame, "Welcome to the program", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "1.Detect the object", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "2. Prediction function", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "3. List of detected object", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
    elif count1 == 20 and initialize == False:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if fingers_folded(hand_landmarks):
                    count4 += 1
                    print(count4)
                    if count4 == 20:
                        count1 = 0
                        initialize = True

            results = model.track(frame, persist=True,conf = 0.8)
            results[0].save_txt("Results.txt")
            frame = results[0].plot()
            # cv2.imshow(window_name, frame)
            #
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #             break
    elif count3 == 20 and initialize == False:
        arr_list = []
        with open("Results.txt", "r") as f:
            for line in f:
                arr = line.split(" ")
                if arr[0] not in arr_list and arr[0] != "0":
                    arr_list.append(arr[0])
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            if lm[16].y < lm[14].y and lm[12].y < lm[10].y and lm[8].y < lm[6].y:
                count5 += 1
                print(count5)
                if count5 == 20:
                    count3 = 0
                    #initialize = True
                    detected_object = data[arr_list[choice]]
                    print(detected_object)
                    count5 = 0
                    processing = True
                    threading.Thread(target=process_model, args=(detected_object,)).start()
            elif lm[12].y < lm[10].y and lm[8].y < lm[6].y and (choice - 1 > -1):
                count6 += 1
                if count6 == 10:
                    choice -= 1
                    count6 = 0

            elif lm[8].y < lm[6].y and (choice + 1 <= len(arr_list) -1):
                count7 += 1
                if count7 == 10:
                    choice += 1
                    count7 = 0


        for i in range(len(arr_list)):
            text = f'{i+1}. {data[arr_list[i]]}'
            if i == choice:
                cv2.putText(frame, text, (10, 100 + i*100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, text, (10, 100 + i * 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow(window_name, frame)
        # count3 = 0
        # initialize = True
    elif processing:
        cv2.putText(frame, "Predicting the price", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
    elif result:
        cv2.putText(frame, f"The price of {detected_object}: ${result:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    time.sleep(0.1)  # Slow down loop iterations slightly.

cap.release()
cv2.destroyAllWindows()











