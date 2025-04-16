import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.patcher import save_patches, extract_images_from_lmdb

root_dir = "datasets/SIDD/train"
train_input_images_folder  = 'SIDD/train/input'
train_target_images_folder = 'SIDD/train/target'

# Make sure target folders exist
os.makedirs(train_input_images_folder, exist_ok=True)
os.makedirs(train_target_images_folder, exist_ok=True)

# Loop through each folder in the root directory
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    # Skip non-folder items
    if not os.path.isdir(folder_path):
        continue

    # Scan inside the folder
    for file_name in os.listdir(folder_path):
        src_file = os.path.join(folder_path, file_name)
        if 'GT' in file_name:
            shutil.move(src_file, train_target_images_folder)
        elif 'NOISY' in file_name:
            shutil.move(src_file, train_input_images_folder)

print("Done moving GT and NOISY files.")

test_input_lmdb_folder  = 'datasets/SIDD/test/input_crops.lmdb/'
test_target_lmdb_folder = 'datasets/SIDD/test/gt_crops.lmdb/'

train_input_patches_folder = "SIDD/train/input_patches"
os.makedirs(train_input_patches_folder, exist_ok=True)

train_target_patches_folder = "SIDD/train/target_patches"
os.makedirs(train_target_patches_folder, exist_ok=True)

test_input_images_folder = "SIDD/test/input"
os.makedirs(test_input_images_folder, exist_ok=True)

test_target_images_folder = "SIDD/test/target"
os.makedirs(test_target_images_folder, exist_ok=True)

test_input_patches_folder = "SIDD/test/input_patches"
os.makedirs(test_input_patches_folder, exist_ok=True)

test_target_patches_folder = "SIDD/test/target_patches"
os.makedirs(test_target_patches_folder, exist_ok=True)

if __name__ == "__main__":
    save_patches(train_input_images_folder, train_input_patches_folder, 'input')
    save_patches(train_target_images_folder, train_target_patches_folder, 'target')

    extract_images_from_lmdb(test_input_lmdb_folder, test_input_images_folder)
    extract_images_from_lmdb(test_target_lmdb_folder, test_target_images_folder)

    save_patches(test_input_images_folder, test_input_patches_folder, 'input')
    save_patches(test_target_images_folder, test_target_patches_folder, 'target')
