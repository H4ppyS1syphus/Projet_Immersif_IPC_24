from helper_tools.dataset import PlantDocDataset
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from helper_tools.engine import train_one_epoch
import helper_tools.utils as utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# Define class names based on the PlantDoc dataset
class_names = [
    "background",  # Index 0 for the background
    "apple_scab", "apple_black_rot", "apple_healthy", 
    "bell_pepper_bacterial_spot", "bell_pepper_healthy",
    "cherry_healthy", "cherry_powdery_mildew",
    "corn_cercospora_leaf_spot_gray_leaf_spot", "corn_common_rust", "corn_healthy", "corn_northern_leaf_blight",
    "grape_black_rot", "grape_healthy", "grape_isariopsis_leaf_spot",
    "peach_bacterial_spot", "peach_healthy", 
    "potato_early_blight", "potato_healthy", "potato_late_blight", 
    "raspberry_healthy",
    "soybean_healthy",
    "squash_powdery_mildew",
    "strawberry_healthy", "strawberry_leaf_scorch",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy", "tomato_late_blight", 
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites", "tomato_target_spot", 
    "tomato_yellow_leaf_curl_virus", "tomato_mosaic_virus"
]

num_classes = len(class_names)  # Background + number of disease categories

# Load the dataset with appropriate transformations
dataset = PlantDocDataset(root='data/dataset-repo/TRAIN', 
                          csv_file='data/dataset-repo/train_labels.csv', 
                          transforms=utils.get_transform(train=True))

dataset_test = PlantDocDataset(root='data/dataset-repo/TEST', 
                               csv_file='data/dataset-repo/test_labels.csv', 
                               transforms=utils.get_transform(train=False))

# Create data loaders for train and test sets
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn
)

# Get the model with the correct number of output classes
model = utils.get_model_instance_segmentation(num_classes)

# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def train_model(model, device, data_loader, data_loader_test, num_epochs=50, lr=0.0001):
    # Move model to the right device (GPU or CPU)
    model.to(device)
    
    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Lists to store metrics
    train_loss_history = []
    lr_history = []
    test_loss_history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Store metrics
        train_loss_history.append(train_loss.meters["loss"].global_avg)  # Store average training loss for the epoch
        lr_history.append(optimizer.param_groups[0]["lr"])  # Store current learning rate
        
        # Update the learning rate
        lr_scheduler.step()
        
        # Evaluate on the test dataset and collect loss
        test_loss = utils.evaluate(model, data_loader_test, device=device)
        test_loss_history.append(test_loss)  # Assuming evaluate returns some loss value
        
        # Save model checkpoint at the end of training
        if epoch == num_epochs - 1:
            print("Saving final model...")
            torch.save(model.state_dict(), "resnet_finetuned_plantdoc_new_20.pth")
    
    # Generate plots after training
    plot_training_metrics(train_loss_history, test_loss_history, lr_history)
    
    print("Training completed.")

def plot_training_metrics(train_loss, test_loss, lr):
    """
    Plots the training loss, test loss, and learning rate across epochs.
    """
    epochs = range(1, len(train_loss) + 1)

    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lr, label="Learning Rate")
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == "__main__":
    train_model(model, device, data_loader, data_loader_test, num_epochs=20, lr=0.0001)
