import cv2
import numpy as np

# Load Haar cascade classifiers
face_classifier = cv2.CascadeClassifier(r"C:\Users\Sriya v\VS CODE\nlp\CNN\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"C:\Users\Sriya v\VS CODE\nlp\CNN\haarcascade_eye.xml")

# Check if classifiers loaded properly
if face_classifier.empty() or eye_classifier.empty():
    print("Error: Could not load Haar cascade XML files.")
    exit()


def face_detector(img, size=0.5):
    """ Detects faces and eyes in a frame and returns the processed image. """

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces are detected, return the original image
    if len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        # Adjust face region slightly for better visibility
        x = max(0, x - 20)
        w = w + 40
        y = max(0, y - 20)
        h = h + 40

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within face ROI
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    return img


# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened properly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply face detector
    processed_frame = face_detector(frame)

    # Show the output
    cv2.imshow("Live Face & Eye Detection", processed_frame)

    # Press 'Enter' (ASCII 13) to exit
    if cv2.waitKey(1) == 13:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
