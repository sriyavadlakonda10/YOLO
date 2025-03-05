import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 640, 480
FPS = 60
BRUSH_SIZE = 10

# Set up the game windows
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw on Face with Finger!")

# Create a second window for arithmetic operations
arithmetic_window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Arithmetic Operations")

# Colors
WHITE = (255, 255, 255)
BRUSH_COLOR = (0, 255, 0)
BLACK = (0, 0, 0)

# Hand Tracking Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Face Detection Setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect hand gestures
def detect_hand_gesture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

# Function to convert hand coordinates to screen coordinates
def hand_to_screen_coordinates(hand_landmarks, frame_width, frame_height):
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_finger_tip.x * frame_width)
        y = int(index_finger_tip.y * frame_height)
        return x, y
    return None, None

# Function to detect faces
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return faces

# Function to perform arithmetic operations
def perform_arithmetic(expression):
    try:
        result = str(eval(expression))
    except:
        result = "Invalid Expression"
    return result

# Main loop
def main():
    cap = cv2.VideoCapture(0)
    clock = pygame.time.Clock()
    last_position = None
    arithmetic_input = ""
    arithmetic_result = ""

    running = True
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the image
        faces = detect_face(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        hand_landmarks = detect_hand_gesture(frame)
        x, y = hand_to_screen_coordinates(hand_landmarks, WIDTH, HEIGHT)

        if x is not None and y is not None:
            if last_position:
                pygame.draw.line(screen, BRUSH_COLOR, last_position, (x, y), BRUSH_SIZE)
            last_position = (x, y)
        else:
            last_position = None  # Reset if no hand detected

        # Convert the frame to a format that can be displayed in Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # Handle arithmetic window
        arithmetic_window.fill(WHITE)
        font = pygame.font.Font(None, 36)
        input_text = font.render(f"Input: {arithmetic_input}", True, BLACK)
        result_text = font.render(f"Result: {arithmetic_result}", True, BLACK)
        arithmetic_window.blit(input_text, (20, 20))
        arithmetic_window.blit(result_text, (20, 60))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    arithmetic_result = perform_arithmetic(arithmetic_input)
                    arithmetic_input = ""
                elif event.key == pygame.K_BACKSPACE:
                    arithmetic_input = arithmetic_input[:-1]
                else:
                    arithmetic_input += event.unicode

        clock.tick(FPS)

    # Clean up
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()