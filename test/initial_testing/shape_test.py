import cv2 as cv
import numpy as np
import time


# Load an image

img = cv.imread('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/laitues.png')

net = cv.dnn.readNetFromDarknet('/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3.cfg', '/home/fred/Documents/projet_immersif/Projet_Immersif_IPC_24/test/initial_testing/network/yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]

text = f'Blob shape={blob.shape}'

net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()
