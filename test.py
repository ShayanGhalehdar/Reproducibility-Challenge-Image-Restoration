import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import argparse
from models.baseline import BaselineModel
from models.nafnet import NAFNetModel
from utils.metrics import psnr_loss, ssim_loss
from utils.dataset import PairedPatchDataset

# Hyperparameters
batch_size = 8
width = 32

if __name__ == "__main__":
    # Prepare Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['Baseline', 'NAFNet'], default='Baseline')
    parser.add_argument('--dataset', type=str, choices=['GoPro', 'SIDD'], default='GoPro')
    args = parser.parse_args()

    test_input_patches_folder = os.path.join(args.dataset, "test", "input_patches")
    test_target_patches_folder = os.path.join(args.dataset, "test", "target_patches")

    test_dataset = PairedPatchDataset(test_input_patches_folder, test_target_patches_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Initialization
    if args.model == 'Baseline':
        model = BaselineModel(3, width).to(device)
        model_tag = 'Baseline'
    else:
        model = NAFNetModel(3, width).to(device)
        model_tag = 'NAFNet'

    model.load_state_dict(torch.load(f"stats/{args.dataset}-{model_tag}/best_model.pth", map_location=device))

    model.eval()

    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_psnr += -psnr_loss(outputs, targets).item()
            total_ssim += ssim_loss(outputs, targets).item()
            count += 1

            if count % 100 == 0:
                print(count)

    test_psnr, test_ssim = total_psnr / count, total_ssim / count

    print('-----------------------------------')
    print(f"Test: PSNR = {test_psnr:.2f} dB | SSIM = {test_ssim:.2f}")