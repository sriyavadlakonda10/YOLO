import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(r"C:\Users\Sriya v\VS CODE\nlp\OBJECT TRACKING\los_angeles.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video Ended or Error in Reading Frame")
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'ESC' to exit
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
