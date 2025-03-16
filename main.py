import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Store circle positions
circle_points = []


# Function to check if the index finger is pointing up
def is_pointing_up(landmarks):
    """Checks if the index finger is pointing up."""
    index_tip = landmarks[8]  # Index finger tip
    index_mcp = landmarks[5]  # Base of index finger
    wrist = landmarks[0]  # Wrist

    # Condition 1: Index tip must be higher than its base
    tip_above_base = index_tip.y < index_mcp.y
    check1 = landmarks[7].y > index_tip.y
    # Condition 2: Index tip should also be higher than the wrist
    #tip_above_wrist = index_tip.y < wrist.y

    #Condition 3 (Optional): The slope between tip and base should be steep
    slope = abs(index_tip.y - index_mcp.y) / abs(index_tip.x - index_mcp.x + 1e-6)  # Avoid division by zero
    steep_angle = slope > 1.5  # Adjust this threshold as needed

    return tip_above_base and check1 and steep_angle
    #return tip_above_base and tip_above_wrist and steep_angle


# Function to detect if only the index finger is pointing up
def is_pointing_up_gesture(landmarks):
    """Checks if the user is making a 'pointing up' gesture"""

    if not is_pointing_up(landmarks):  # Ensure index is pointing up
        return False

        # Other fingers
    middle_tip, middle_mcp = landmarks[12], landmarks[9]
    ring_tip, ring_mcp = landmarks[16], landmarks[13]
    pinky_tip, pinky_mcp = landmarks[20], landmarks[17]

    # Ensure other fingers are folded (tip below base)
    fingers_folded = (
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
    )

    return fingers_folded  # True if pointing up with only the index finger


# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB (MediaPipe requirement)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance
        results = hands.process(image)  # Detect hands

        # Convert image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get image size
        h, w, _ = image.shape

        gesture_text = ""  # Initialize text

        # Draw detected hands and process landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmarks = hand_landmarks.landmark  # Get landmarks

                # Check if the user is "pointing up"
                if is_pointing_up_gesture(landmarks):
                    gesture_text = "☝️ Pointing Up!"
                    x, y = landmarks[8].x, landmarks[8].y  # Get index finger tip position
                    x, y = int(x * w), int(y * h)  # Convert to pixel coordinates
                    circle_points.append((x, y))  # Store points for drawing circles


        # Draw circles at stored points
        for point in circle_points:
            cv2.circle(image, point, 10, (255, 0, 255), -1)  # Draw filled circle

        # Display gesture text on screen
        if gesture_text:
            cv2.putText(image, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with hand tracking and drawing
        cv2.imshow('Pointing Up Drawing', cv2.flip(image, 1))  # Flip for mirror effect

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
