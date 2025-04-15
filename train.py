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

def set_seed(seed=25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def validate():
    print("---------------------------------------")
    print("Validating ...")
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_psnr += -psnr_loss(outputs, targets).item()
            total_ssim += ssim_loss(outputs, targets).item()
            count += 1
    model.train()
    return total_psnr / count, total_ssim / count

set_seed(25)  # Set a fixed seed for reproducibility

# Hyperparameters
batch_size = 8
num_iterations = 100000
learning_rate = 1e-3
min_lr = 1e-6
width = 32
val_split = 0.05  # 5% for validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['Baseline', 'NAFNet'], default='Baseline')
    parser.add_argument('--dataset', type=str, choices=['GoPro', 'SIDD'], default='GoPro')
    args = parser.parse_args()

    train_input_patches_folder = os.path.join(args.dataset, "train", "input_patches")
    train_target_patches_folder = os.path.join(args.dataset, "train", "target_patches")

    dataset = PairedPatchDataset(train_input_patches_folder, train_target_patches_folder)

    # Split dataset into training and validation
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Initialization
    if args.model == 'Baseline':
        model = BaselineModel(3, width).to(device)
        model_tag = 'Baseline'
    else:
        model = NAFNetModel(3, width).to(device)
        model_tag = 'NAFNet'

    os.makedirs(f'stats/{args.dataset}-{model_tag}/checkpoints', exist_ok=True)

    # Enable Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()  # Prevents FP16 underflow
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.9), weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=min_lr)
    
    # TensorBoard Writer
    writer = SummaryWriter()

    best_psnr = float('-inf')

    # Training Loop
    model.train()
    iteration = 0
    while iteration < num_iterations:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
    
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Mixed precision
                outputs = model(inputs)
                loss = psnr_loss(outputs, targets)

            ssim = ssim_loss(outputs, targets)
            scaler.scale(loss).backward()  # Scale gradients to avoid FP16 underflow

            # Gradient clipping
            scaler.unscale_(optimizer)  # Unscales gradients before clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if iteration % 100 == 0:
                psnr_value = -loss.item()
                writer.add_scalar('Train/PSNR', psnr_value, iteration)
                writer.add_scalar('Train/SSIM', ssim.item(), iteration)
                print(f"Iteration {iteration}: PSNR = {psnr_value:.2f} dB | SSIM = {ssim:.2f}")

            if iteration %  1000 == 0:
                val_psnr, val_ssim = validate()
                writer.add_scalar('Validation/PSNR', val_psnr, iteration)
                writer.add_scalar('Validation/SSIM', val_ssim, iteration)
                print(f"Validation: PSNR = {val_psnr:.2f} dB | SSIM = {val_ssim:.2f}")
                torch.save(model.state_dict(), f"stats/{args.dataset}-{model_tag}/checkpoints/checkpoint_{iteration}.pth")
                print("Checkpoint saved!")

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(model.state_dict(), f"stats/{args.dataset}-{model_tag}/best_model.pth")
                    print("Best Model Saved!")

                print("---------------------------------------")

            iteration += 1
            if iteration >= num_iterations:
                break

    torch.save({
        'iteration': 100000,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item()
    },f'stats/{args.dataset}-{model_tag}/checkpoint_100000.pth')

    print("Training complete.")
    writer.close()