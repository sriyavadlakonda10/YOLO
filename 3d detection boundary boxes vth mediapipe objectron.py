# --------------> IMPORT LIBRARY
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import urllib.request  # Fixed import

# Initialize MediaPipe Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils  # Helper function

# -----------> Function to Read Image from URL
def url_to_array(url):
    req = urllib.request.urlopen(url)
    arr = np.array(bytearray(req.read()), dtype=np.uint8)  # Fixed dtype
    arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr

# -----------> Load Image (Local File)
mug_path = r"D:\Download\reina-lovefull-WKEEyDKJ8ak-unsplash.jpg"  # Fixed path
mug = cv2.imread(mug_path)  # Load local image
mug = cv2.cvtColor(mug, cv2.COLOR_BGR2RGB)  # Convert to RGB

# ------> Instantiate Objectron
objectron = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.2,
    model_name='Cup'
)

# Inference
results = objectron.process(mug)

#--------------> Display the Result
if not results.detected_objects:
    print(f'No box landmarks detected.')

# Copy image so as not to draw on the original one
annotated_image = mug.copy()
for detected_object in results.detected_objects:
    # Draw landmarks
    mp_drawing.draw_landmarks(
        annotated_image,
        detected_object.landmarks_2d,
        mp_objectron.BOX_CONNECTIONS
    )

    # Draw axis
    mp_drawing.draw_axis(
        annotated_image,
        detected_object.rotation,
        detected_object.translation
    )

# -----------> Plot the Result
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(annotated_image)
ax.axis('off')
plt.show()
