import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load class names from COCO dataset
with open(r"C:\Users\Sriya v\VS CODE\nlp\YOLO\utils\coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load a pretrained YOLOv8 model
model = YOLO(r"C:\Users\Sriya v\VS CODE\nlp\YOLO\weights\yolov8n.pt")  # Ensure correct .pt file

# Video Capture
cap = cv2.VideoCapture(r"C:\Users\Sriya v\VS CODE\nlp\YOLO\video\video_sample1.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Perform inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            clsID = int(box.cls[0].item())  # Class ID
            conf = float(box.conf[0].item())  # Confidence score
            bb = box.xyxy[0].cpu().numpy()  # Bounding box coordinates

            # Draw bounding box
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 3)

            # Display class name and confidence
            label = f"{class_list[clsID]}: {conf:.2f}"
            cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
