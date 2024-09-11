import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision import transforms as T
import xlrd
import csv
from os import sys
import glob

class PlantDocDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load the CSV file containing annotations
        self.csv_data = pd.read_csv(csv_file)
        # Get the image filenames from the CSV
        self.imgs = self.csv_data["filename"].unique()

    def __getitem__(self, idx):
        # Get image file name and full path
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        img = read_image(img_path)

        # Extract annotations for this image
        records = self.csv_data[self.csv_data["filename"] == img_name]

        # Create the bounding boxes
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Assume all labels are 1 for plant disease (or you can modify if you have more classes)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

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


