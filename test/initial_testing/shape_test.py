import cv2 as cv
import numpy as np
import time

# Load an image
img = cv.imread('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/rue.png')
# Load names of classes
classes = open('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/coco.names').read().strip().split('\n')
print(classes)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Load YOLOv3-tiny model
net = cv.dnn.readNetFromDarknet('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3-tiny.cfg', '/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# Determine the output layer
ln = net.getLayerNames()

# Construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

t0 = time.time()
outputs = net.forward(ln)
t = time.time()
print('Forward propagation time:', t-t0)

# Prepare for bounding box drawing
boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

for output in outputs:
    print("Output shape:", output.shape)  # Debugging output shapes
    for detection in output:
        scores = detection[5:]
        print(scores)
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply Non-Max Suppression
indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Specify the path and filename where you want to save the image
output_image_path = 'result_image.png'

# Save the image
cv.imwrite(output_image_path, img)

print("Image saved to", output_image_path)
