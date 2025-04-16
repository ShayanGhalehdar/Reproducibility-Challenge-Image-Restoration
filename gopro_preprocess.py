import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.patcher import save_patches, extract_images_from_lmdb

train_input_images_folder  = 'datasets/GoPro/train/input'
train_target_images_folder = 'datasets/GoPro/train/target'
test_input_lmdb_folder  = 'datasets/GoPro/test/input.lmdb/'
test_target_lmdb_folder = 'datasets/GoPro/test/target.lmdb/'

train_input_patches_folder = "GoPro/train/input_patches"
os.makedirs(train_input_patches_folder, exist_ok=True)

train_target_patches_folder = "GoPro/train/target_patches"
os.makedirs(train_target_patches_folder, exist_ok=True)

test_input_images_folder = "GoPro/test/input"
os.makedirs(test_input_images_folder, exist_ok=True)

test_target_images_folder = "GoPro/test/target"
os.makedirs(test_target_images_folder, exist_ok=True)

test_input_patches_folder = "GoPro/test/input_patches"
os.makedirs(test_input_patches_folder, exist_ok=True)

test_target_patches_folder = "GoPro/test/target_patches"
os.makedirs(test_target_patches_folder, exist_ok=True)

if __name__ == "__main__":
    save_patches(train_input_images_folder, train_input_patches_folder, 'input')
    save_patches(train_target_images_folder, train_target_patches_folder, 'target')

    extract_images_from_lmdb(test_input_lmdb_folder, test_input_images_folder)
    extract_images_from_lmdb(test_target_lmdb_folder, test_target_images_folder)

    save_patches(test_input_images_folder, test_input_patches_folder, 'input')
    save_patches(test_target_images_folder, test_target_patches_folder, 'target')