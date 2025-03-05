import numpy as np
import cv2

# Load the Haar cascade classifiers
face_classifier = cv2.CascadeClassifier(r"C:\Users\Sriya v\VS CODE\nlp\CNN\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"C:\Users\Sriya v\VS CODE\nlp\CNN\haarcascade_eye.xml")

# Verify that the classifiers are loaded correctly
if face_classifier.empty() or eye_classifier.empty():
    print("Error: Could not load Haar cascade XML files.")
    exit()

# Load the image
image_path = r"C:\Users\Sriya v\OneDrive\Documents\Pictures\sriya.jpg1.jpg"
img = cv2.imread(image_path)

# Check if image is loaded properly
if img is None:
    print("Error: Could not read the image file.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# If no faces are found, print message and exit
if len(faces) == 0:
    print("No Face Found.")
else:
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Extract face region for eye detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    # Display the final image after drawing rectangles
    cv2.imshow("Face and Eye Detection", img)
    cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()