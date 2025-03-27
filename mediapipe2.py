import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe for face and hand detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Load Filters (Ensure PNGs have transparency)
filters = {
    "sunglasses": cv2.imread(r"C:\toxic  kmax\sun.png", cv2.IMREAD_UNCHANGED),
    "mustache": cv2.imread(r"C:\toxic  kmax\mustache1.png", cv2.IMREAD_UNCHANGED),
    "dog_filter": cv2.imread(r"C:\Users\divya\Downloads\5a5bc0b914d8c4188e0b0951.png", cv2.IMREAD_UNCHANGED),
}

filter_keys = list(filters.keys())  # ['sunglasses', 'mustache', 'dog_filter']
current_filter_idx = 0  # Default filter index

# Start Webcam Capture
cap = cv2.VideoCapture(0)

def overlay_filter(frame, filter_img, x, y, w, h):
    """ Overlays a PNG filter on the frame at (x, y) with width w and height h """
    filter_resized = cv2.resize(filter_img, (w, h))
    alpha_s = filter_resized[:, :, 3] / 255.0  # Extract alpha channel for transparency
    alpha_l = 1.0 - alpha_s

    for c in range(3):  # Apply overlay per color channel
        frame[y:y+h, x:x+w, c] = (alpha_s * filter_resized[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])

def count_fingers(hand_landmarks):
    """ Counts raised fingers based on hand landmarks """
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4
    fingers = 0

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Finger raised condition
            fingers += 1

    # Thumb condition
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:  
        fingers += 1

    return fingers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh Detection
    result_face = face_mesh.process(rgb_frame)
    
    # Hand Tracking
    result_hands = hands.process(rgb_frame)

    # Change Filter Based on Hand Gesture
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            fingers_count = count_fingers(hand_landmarks)

            if fingers_count == 1:
                current_filter_idx = 0  # Sunglasses
            elif fingers_count == 2:
                current_filter_idx = 1  # Mustache
            elif fingers_count == 5:
                current_filter_idx = 2  # Dog Filter

    # Get the current filter
    current_filter = filters[filter_keys[current_filter_idx]]

    # Apply Filter on Face
    if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:
            # Get key facial landmarks
            left_cheek_x = int(face_landmarks.landmark[234].x * w)  # Left cheek
            right_cheek_x = int(face_landmarks.landmark[454].x * w)  # Right cheek
            chin_y = int(face_landmarks.landmark[152].y * h)  # Chin
            forehead_y = int(face_landmarks.landmark[10].y * h)  # Forehead

            face_width = right_cheek_x - left_cheek_x
            face_height = chin_y - forehead_y

            # Apply Mustache
            if filter_keys[current_filter_idx] == "mustache":
                nose_x = int(face_landmarks.landmark[1].x * w)  # Nose center
                nose_y = int(face_landmarks.landmark[1].y * h)  # Nose bottom

                mustache_width = int(face_width * 0.7)
                mustache_height = int(face_height * 0.2)
                mustache_x = nose_x - mustache_width // 2
                mustache_y = nose_y + int(face_height * 0.05)

                overlay_filter(frame, current_filter, mustache_x, mustache_y, mustache_width, mustache_height)

            # Apply Sunglasses
            elif filter_keys[current_filter_idx] == "sunglasses":
                left_eye_x = int(face_landmarks.landmark[33].x * w)  # Left Eye
                right_eye_x = int(face_landmarks.landmark[263].x * w)  # Right Eye
                eye_y = int(face_landmarks.landmark[168].y * h)  # Eye Center

                sunglasses_width = face_width
                sunglasses_height = int(face_height * 0.3)
                sunglasses_x = left_eye_x - int(face_width * 0.1)
                sunglasses_y = eye_y - int(face_height * 0.2)

                overlay_filter(frame, current_filter, sunglasses_x, sunglasses_y, sunglasses_width, sunglasses_height)

            # Apply Dog Filter (Ears + Nose)
            elif filter_keys[current_filter_idx] == "dog_filter":
                nose_x = int(face_landmarks.landmark[1].x * w)  # Nose center
                nose_y = int(face_landmarks.landmark[1].y * h)  # Nose bottom
                forehead_y = int(face_landmarks.landmark[10].y * h)  # Forehead
                chin_y = int(face_landmarks.landmark[152].y * h)  # Chin
                left_cheek_x = int(face_landmarks.landmark[234].x * w)  # Left cheek
                right_cheek_x = int(face_landmarks.landmark[454].x * w)  # Right cheek

                # Calculate the face height and width
                face_width = right_cheek_x - left_cheek_x
                face_height = chin_y - forehead_y

                # Adjust dog filter size based on face width and height
                dog_filter_width = int(face_width * 1.2)
                dog_filter_height = int(face_height * 1.3)  # Increased height to fit ears and nose

                # Position the dog filter so that the nose part aligns with the actual nose
                  # Shift upwards for ears
                  

                # Overlay the full dog filter (ear + nose)
                overlay_filter(frame, current_filter, dog_filter_x, dog_filter_y, dog_filter_width, dog_filter_height)

    # Display the Frame
    cv2.putText(frame, f"Filter: {filter_keys[current_filter_idx].capitalize()}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Snapchat Filters with Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
