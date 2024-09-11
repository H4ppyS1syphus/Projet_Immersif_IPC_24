from helper_tools.dataset import PlantDocDataset
import torch
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from helper_tools.engine import train_one_epoch, evaluate
import helper_tools.utils as utils
# Define transforms


# Load the dataset
dataset = PlantDocDataset(root='data/dataset-repo/TRAIN', 
                          csv_file='data/dataset-repo/train_labels.csv', 
                          transforms=utils.get_transform(train=True))

dataset_test = PlantDocDataset(root='data/dataset-repo/TEST', 
                               csv_file='data/dataset-repo/test_labels.csv', 
                               transforms=utils.get_transform(train=False))

# Create data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn
)


# Define number of classes (background + 1 for plant disease or more if you have multiple classes)
num_classes = 2  # Adjust this based on the number of plant disease categories

# Get the model with the correct number of output classes
model = utils.get_model_instance_segmentation(num_classes)

# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def train_model(model, device, data_loader, data_loader_test, num_epochs=10, lr=0.005):
    # Move model to the right device (GPU or CPU)
    model.to(device)
    
    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Update the learning rate
        lr_scheduler.step()
        
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), "resnet_finetuned_leaf_classification.pth")
    
    print("Training completed.")

if __name__ == "__main__":
    train_model(model, device, data_loader, data_loader_test, num_epochs=10, lr=0.005)
    # Evaluate the model on the test set
    evaluate(model, data_loader_test, device=device)
