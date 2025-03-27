import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Initialize Game Window
WIDTH, HEIGHT = 640, 480
paddle_width, paddle_height = 100, 20
ball_radius = 10

# Ball Properties
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 5, 5

# Paddle Properties
paddle_x = WIDTH // 2 - paddle_width // 2
paddle_y = HEIGHT - 50

# Score
score = 0

# Capture Video from Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand Detection
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
            paddle_x = x - paddle_width // 2  # Move paddle based on wrist position

    # Keep Paddle Inside Bounds
    paddle_x = max(0, min(WIDTH - paddle_width, paddle_x))

    # Move Ball
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball Collision with Walls
    if ball_x <= 0 or ball_x >= WIDTH - ball_radius:
        ball_dx = -ball_dx

    if ball_y <= 0:
        ball_dy = -ball_dy

    # Ball Collision with Paddle
    if paddle_y < ball_y + ball_radius < paddle_y + paddle_height and paddle_x < ball_x < paddle_x + paddle_width:
        ball_dy = -ball_dy
        score += 1

    # Ball Falls Below the Paddle (Game Over Condition)
    if ball_y > HEIGHT:
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2  # Reset Ball
        score = 0  # Reset Score

    # Draw Paddle
    cv2.rectangle(frame, (paddle_x, paddle_y), (paddle_x + paddle_width, paddle_y + paddle_height), (0, 255, 0), -1)

    # Draw Ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Draw Score
    cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show Frame
    cv2.imshow("Hand Gesture Pong", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
