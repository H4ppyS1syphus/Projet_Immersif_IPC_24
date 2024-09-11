from skimage import io,transform
import numpy as np
import os

def normalize_image(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.uint8(normalized_image * 255)

def redimImage640(dossier_images, dossier_images_redim):

    if os.path.exists(dossier_images_redim):
        compteur = 1
        while os.path.exists(f"{dossier_images_redim}_{compteur}"):
            compteur += 1
        dossier_images_redim = f"{dossier_images_redim}_{compteur}"
        os.makedirs(dossier_images_redim)
    else:
        os.makedirs(dossier_images_redim)
    
    for image in os.listdir(dossier_images):
        chemin_image = os.path.join(dossier_images, image)
        image_init = io.imread(chemin_image)

        hauteur, largeur, canaux = image_init.shape
        echelle = 640 / max(hauteur, largeur)
        image_redim = transform.resize(image_init, (int(hauteur*echelle), int(largeur*echelle)))
        chemin_image_redim = os.path.join(dossier_images_redim, image)

        io.imsave(chemin_image_redim, normalize_image(image_redim))

# Exemple
# redimImage640(r"C:\Users\bapti\Desktop\Projet_Immersif_IPC_24\testImage", r"C:\Users\bapti\Desktop\Projet_Immersif_IPC_24\testImageRedim")
