import pandas as pd
import os

def creerDictionnaire(chemin_csv, chemin_images):

    dictionnaire = {}

    if chemin_images:
        liste_image = os.listdir(chemin_images)
    
    if chemin_csv:
        csv = pd.read_csv(chemin_csv)
    
    for i in range (len(csv)):
        image = csv.iloc[i]["filename"]

        if image in liste_image:
            coordonnees = [int(csv.iloc[i]["xmin"]), int(csv.iloc[i]["ymin"]), int(csv.iloc[i]["xmax"]), int(csv.iloc[i]["ymax"])]
        
        if image in dictionnaire:
            dictionnaire[image].append(coordonnees)
        else:
            dictionnaire[image] = [coordonnees]

    return dictionnaire


# EXEMPLE
# creerDictionnaire(r"C:\Users\bapti\Desktop\PlantDoc-Object-Detection-Dataset-master\test_labels.csv", r"C:\Users\bapti\Desktop\PlantDoc-Object-Detection-Dataset-master\TEST")


