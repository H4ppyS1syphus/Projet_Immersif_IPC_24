import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# 1. Charger le modèle sauvegardé
num_classes = 2  # Deux classes : sain et malade
model = models.resnet18(pretrained=False)  # On ne charge pas les poids pré-entraînés
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# Charger les poids du modèle entraîné
model.load_state_dict(torch.load("disease_detection/resnet_finetuned_leaf_classification.pth", map_location=torch.device('cpu')))
# Mettre le modèle en mode évaluation
model.eval()

# 2. Préparer l'image
# Chemin de la nouvelle image
image_path = "disease_detection/test.jpg"

# Définir les transformations (doivent être identiques à celles utilisées lors de l'entraînement)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Charger et transformer l'image
image = Image.open(image_path)
image = data_transforms(image)

# Ajouter une dimension batch (car le modèle attend un batch d'images)
image = image.unsqueeze(0)

# 3. Faire la prédiction
with torch.no_grad():  # Pas besoin de calculer les gradients pour l'inférence
    outputs = model(image)
    _, preds = torch.max(outputs, 1)  # Obtenir l'indice de la classe prédite

# Classes prédéfinies (sain ou malade)
classes = ['saine', 'malade']

# Afficher le résultat
print(f"La feuille est {classes[preds[0]]}.")
