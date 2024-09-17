import cv2 as cv
import numpy as np
import time

# Load an image
img = cv.imread('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/laitues.png')
# Load names of classes
classes = open('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Load YOLOv3-tiny model
net = cv.dnn.readNetFromDarknet('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3-tiny.cfg', '/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# Determine the output layer
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]# Prepare the image blob

blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Perform the forward pass
t0 = time.time()
outputs = net.forward(ln)
t = time.time()
print('Forward propagation time:', t-t0)

# Process the outputs
boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

for output in outputs:
    for detection in output:
        # Extract the scores, followed by the class probability
        scores = detection[5:]  # Starting from index 5 to skip box coordinates and objectness score
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > 0.5:
            box = detection[0:4] * np.array([w, h, w, h])
            centerX, centerY, width, height = box.astype('int')
            x = int(centerX - width / 2)
            y = int(centerY - height / 2)

            # Populate our lists for NMS
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Non-max suppression
indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the bounding boxes
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{classes[classIDs[i]]}: {confidences[i]:.2f}"
        cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Save the image
cv.imwrite('detected_output.jpg', img)
print("Image saved as 'detected_output.jpg'")

