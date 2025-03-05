import cv2

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\Sriya v\VS CODE\nlp\OBJECT TRACKING\los_angeles.mp4")

# Check if video loaded properly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Background subtractor for object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    # Apply object detection
    mask = object_detector.apply(frame)

    # Display output
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Exit if ESC key is pressed
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
