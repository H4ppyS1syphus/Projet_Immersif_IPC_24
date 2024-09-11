from skimage import io,transform
import numpy as np
import os


def normalize_image(image):
    """
    Normalise l'image pour qu'elle ait des valeurs dans l'intervalle [0, 255] et la convertit en uint8.
    
    Paramètres :
        image : np.array
            L'image d'entrée avec des valeurs dans l'intervalle [-1, 1].
    
    Retour :
        np.array
            L'image normalisée avec des valeurs dans l'intervalle [0, 255] et de type uint8.
    """
    # Normaliser l'image pour qu'elle ait des valeurs dans l'intervalle [0, 1]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Étendre les valeurs pour qu'elles soient dans l'intervalle [0, 255] et convertir en uint8
    return np.uint8(normalized_image * 255)

def redimImage640_640(dossier_images, dossier_images_redim):

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

        image_redim = transform.resize(image_init, (640,640))
        chemin_image_redim = os.path.join(dossier_images_redim, image)

        io.imsave(chemin_image_redim, normalize_image(image_redim))

# Exemple
# redimImage640_640(r"C:\Users\bapti\Desktop\Projet_Immersif_IPC_24\testImage", r"C:\Users\bapti\Desktop\Projet_Immersif_IPC_24\testImageRedim")
