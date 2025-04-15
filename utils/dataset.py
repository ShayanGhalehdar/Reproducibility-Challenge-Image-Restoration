import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

class PairedPatchDataset(Dataset):
    def __init__(self, input_folder, target_folder):
        self.input_folder = input_folder
        self.target_folder = target_folder

        input_files = {f.name.replace("input_", "").split('.')[0] for f in os.scandir(input_folder) if f.is_file()}
        target_files = {f.name.replace("target_", "").split('.')[0] for f in os.scandir(target_folder) if f.is_file()}

        # Find common filenames
        self.image_filenames = sorted(input_files & target_files)

        self.input_paths = [os.path.join(input_folder, f"input_{name}.png") for name in self.image_filenames]
        self.target_paths = [os.path.join(target_folder, f"target_{name}.png") for name in self.image_filenames]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_paths[idx]).convert("RGB")
        target_image = Image.open(self.target_paths[idx]).convert("RGB")

        return self.transform(input_image), self.transform(target_image)