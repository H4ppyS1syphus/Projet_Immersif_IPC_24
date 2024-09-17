import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# 1. Préparation des données
data_dir = "./all"  # dossier contenant les dossiers 'sick' et 'healthy'

# Les transformations pour redimensionner, normaliser, et appliquer les augmentations d'images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to check if a file is a valid image
class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# Function to check if a file is a valid image
def is_valid_file(filepath):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    return os.path.splitext(filepath)[1].lower() in valid_extensions

# Use CustomImageFolder for dataset loading
image_datasets = {x: CustomImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x], is_valid_file=is_valid_file) 
                  for x in ['train', 'val']}

# Création des DataLoader
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) 
               for x in ['train', 'val']}

# Nombre de classes (sain, malade)
num_classes = 2

# 2. Charger ResNet18 pré-entraîné
model = models.resnet18(pretrained=True)

# 3. Modifier la dernière couche (Fully connected) pour la classification binaire
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Vous n'avez pas besoin d'envoyer le modèle sur GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# 4. Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

# Fonction pour l'entraînement
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    # Lists to store the general loss and accuracy
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Lists to store loss and accuracy for Healthy and Sick
    train_loss_healthy = []
    train_loss_sick = []
    train_acc_healthy = []
    train_acc_sick = []

    val_loss_healthy = []
    val_loss_sick = []
    val_acc_healthy = []
    val_acc_sick = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Separate loss and correct counters for Healthy (label 0) and Sick (label 1)
            running_loss_healthy = 0.0
            running_loss_sick = 0.0
            running_corrects_healthy = 0
            running_corrects_sick = 0
            num_healthy = 0
            num_sick = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward and backward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics for general loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Class-specific statistics
                for i in range(len(labels)):
                    if labels[i] == 0:  # Healthy
                        running_loss_healthy += loss.item()
                        running_corrects_healthy += (preds[i] == labels[i]).item()
                        num_healthy += 1
                    else:  # Sick
                        running_loss_sick += loss.item()
                        running_corrects_sick += (preds[i] == labels[i]).item()
                        num_sick += 1

            # Calculate epoch loss and accuracy for the current phase
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Calculate epoch loss and accuracy for Healthy and Sick classes
            if num_healthy > 0:
                epoch_loss_healthy = running_loss_healthy / num_healthy
                epoch_acc_healthy = running_corrects_healthy / num_healthy
            else:
                epoch_loss_healthy = 0
                epoch_acc_healthy = 0

            if num_sick > 0:
                epoch_loss_sick = running_loss_sick / num_sick
                epoch_acc_sick = running_corrects_sick / num_sick
            else:
                epoch_loss_sick = 0
                epoch_acc_sick = 0

            # Print the statistics
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Healthy Loss: {epoch_loss_healthy:.4f} Acc: {epoch_acc_healthy:.4f}')
            print(f'{phase} Sick Loss: {epoch_loss_sick:.4f} Acc: {epoch_acc_sick:.4f}')
            
            # Append the statistics to the history lists
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_loss_healthy.append(epoch_loss_healthy)
                train_loss_sick.append(epoch_loss_sick)
                train_acc_healthy.append(epoch_acc_healthy)
                train_acc_sick.append(epoch_acc_sick)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_loss_healthy.append(epoch_loss_healthy)
                val_loss_sick.append(epoch_loss_sick)
                val_acc_healthy.append(epoch_acc_healthy)
                val_acc_sick.append(epoch_acc_sick)

    # Return the model and the recorded statistics
    return model, {
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
        'train_loss_healthy': train_loss_healthy,
        'train_loss_sick': train_loss_sick,
        'train_acc_healthy': train_acc_healthy,
        'train_acc_sick': train_acc_sick,
        'val_loss_healthy': val_loss_healthy,
        'val_loss_sick': val_loss_sick,
        'val_acc_healthy': val_acc_healthy,
        'val_acc_sick': val_acc_sick
    }



# 5. Entraîner le modèle
model, history = train_model(model, criterion, optimizer, dataloaders, num_epochs=10)

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "resnet_finetuned_leaf_classification.pth")

# Function to plot the data


# Function to convert lists of tensors to lists of CPU-based NumPy arrays
def convert_tensor_list_to_numpy(tensor_list):
    try: 
        return [tensor.cpu().numpy() for tensor in tensor_list]
    except:
        return tensor_list

# Function to plot the data
def plot_metrics(history, num_epochs):
    print("ha")
    epochs = range(1, num_epochs + 1)

    # Convert tensor lists to NumPy arrays
    train_acc_history = convert_tensor_list_to_numpy(history['train_acc_history'])
    val_acc_history = convert_tensor_list_to_numpy(history['val_acc_history'])
    
    train_loss_history = convert_tensor_list_to_numpy(history['train_loss_history'])
    val_loss_history = convert_tensor_list_to_numpy(history['val_loss_history'])

    train_loss_healthy = convert_tensor_list_to_numpy(history['train_loss_healthy'])
    train_loss_sick = convert_tensor_list_to_numpy(history['train_loss_sick'])

    train_acc_healthy = convert_tensor_list_to_numpy(history['train_acc_healthy'])
    train_acc_sick = convert_tensor_list_to_numpy(history['train_acc_sick'])

    val_loss_healthy = convert_tensor_list_to_numpy(history['val_loss_healthy'])
    val_loss_sick = convert_tensor_list_to_numpy(history['val_loss_sick'])

    val_acc_healthy = convert_tensor_list_to_numpy(history['val_acc_healthy'])
    val_acc_sick = convert_tensor_list_to_numpy(history['val_acc_sick'])
    
    print(train_acc_history)
    # Plot general loss
    plt.figure()
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # Plot general accuracy
    plt.figure()
    plt.plot(epochs, train_acc_history, label='Train Accuracy')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    # Plot Healthy vs Sick Loss during training
    plt.figure()
    plt.plot(epochs, train_loss_healthy, label='Train Healthy Loss')
    plt.plot(epochs, train_loss_sick, label='Train Sick Loss')
    plt.title('Train Healthy and Sick Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_healthy_vs_sick_loss.png')
    plt.close()

    # Plot Healthy vs Sick Accuracy during training
    plt.figure()
    plt.plot(epochs, train_acc_healthy, label='Train Healthy Accuracy')
    plt.plot(epochs, train_acc_sick, label='Train Sick Accuracy')
    plt.title('Train Healthy and Sick Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('train_healthy_vs_sick_acc.png')
    plt.close()

    # Plot Healthy vs Sick Loss during validation
    plt.figure()
    plt.plot(epochs, val_loss_healthy, label='Validation Healthy Loss')
    plt.plot(epochs, val_loss_sick, label='Validation Sick Loss')
    plt.title('Validation Healthy and Sick Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('val_healthy_vs_sick_loss.png')
    plt.close()

    # Plot Healthy vs Sick Accuracy during validation
    plt.figure()
    plt.plot(epochs, val_acc_healthy, label='Validation Healthy Accuracy')
    plt.plot(epochs, val_acc_sick, label='Validation Sick Accuracy')
    plt.title('Validation Healthy and Sick Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('val_healthy_vs_sick_acc.png')
    plt.close()

# Call the plot_metrics function to generate and save all the plots
plot_metrics(history, num_epochs=10)
