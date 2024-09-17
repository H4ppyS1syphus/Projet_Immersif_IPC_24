import matplotlib.pyplot as plt
from skimage import io, color, morphology
import numpy as np


def get_image(path):
    return io.imread(path)

def convert_rgb2hsv(image_rgb):
    image_rgb_np = np.array(image_rgb)
    image_hsv = color.rgb2hsv(image_rgb_np)
    return image_hsv


def create_mask(image_hsv, lower_green, upper_green):
    mask = (image_hsv[:, :, 0] >= lower_green[0]) & (image_hsv[:, :, 0] <= upper_green[0]) & \
        (image_hsv[:, :, 1] >= lower_green[1]) & (image_hsv[:, :, 1] <= upper_green[1]) & \
        (image_hsv[:, :, 2] >= lower_green[2]) & (image_hsv[:, :, 2] <= upper_green[2])
    return mask

def print_raw_and_extracted(imag_raw, extracted_extracted):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(imag_raw)
    ax[0].set_title("Image originale")
    ax[0].axis('off')

    ax[1].imshow(extracted_extracted)
    ax[1].set_title("Image après extraction")
    ax[1].axis('off')

    plt.show()



# On défini des seuils de verts pour l'extraction de contours
       # A termes il faut trouver un moyen pour ces seuils se règlent automatiquement
lower_green = np.array([0.08, 0.08, 0.08])
upper_green = np.array([0.5, 1.0, 1.0])
