import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os
from PIL import Image
import numpy as np

# 1. Define the data directory and transformations
data_dir = "/PlantVillage-Dataset/PlantVillage-Dataset/raw/segmented/"  # Root directory containing 'test' folder
test_dir = os.path.join(data_dir, 'all')

# Ensure the test directory exists
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test directory '{test_dir}' not found.")

# Define the same data transformations used for validation
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Function to check if a file is a valid image
def is_valid_file(filepath):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    return os.path.splitext(filepath)[1].lower() in valid_extensions

# Custom ImageFolder to exclude hidden files/directories
class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# 2. Load the test dataset
test_dataset = CustomImageFolder(root=test_dir, transform=test_transforms, is_valid_file=is_valid_file)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 3. Load the trained model
num_classes = 2  # Assuming 'healthy' and 'sick' classes
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Load the saved model weights
model_path = "disease_detection/resnet_finetuned_leaf_classification.pth"
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# 4. Iterate over the test dataset and collect predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
conf_matrix = confusion_matrix(all_labels, all_preds)

# 6. Print out the metrics
print("Evaluation Metrics on Test Dataset:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
