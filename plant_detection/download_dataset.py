import subprocess
import os

def download_dataset():
    # URL of the external Git repository containing the dataset
    dataset_repo_url = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset"
    
    # Path where you want to store the dataset within your project
    dataset_dir = "data/dataset-repo"

    # Check if dataset directory already exists
    if not os.path.exists(dataset_dir):
        print("Dataset not found. Downloading...")
        # Clone the dataset repository
        subprocess.run(["git", "clone", dataset_repo_url, dataset_dir], check=True)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists. Skipping download.")

if __name__ == "__main__":
    download_dataset()
