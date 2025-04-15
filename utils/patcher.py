import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import lmdb
import cv2

def compute_stride(image_size, patch_size):
    """
    Compute the optimal stride to ensure full coverage while maximizing uniform patches.
    """
    num_patches = (image_size + patch_size - 1) // patch_size  # Equivalent to math.ceil
    return (image_size - patch_size) // (num_patches - 1) if num_patches > 1 else 1


def extract_patches(image, patch_size=256):
    """
    Extracts patches ensuring full coverage.
    """
    _, h, w = image.shape
    stride_h = compute_stride(h, patch_size)
    stride_w = compute_stride(w, patch_size)

    # Patch indices
    i_vals = torch.arange(0, h - patch_size + 1, stride_h)
    j_vals = torch.arange(0, w - patch_size + 1, stride_w)

    # Number of Patches
    num_patches_h = i_vals.shape[0]
    num_patches_w = j_vals.shape[0]

    # Preallocate tensor for patches
    patches = torch.empty((num_patches_h * num_patches_w, 3, patch_size, patch_size), dtype=image.dtype, device=image.device)

    indices = []
    patch_idx = 0
    for i in i_vals:
        for j in j_vals:
            patches[patch_idx] = image[:, i:i+patch_size, j:j+patch_size]
            indices.append((i.item(), j.item()))
            patch_idx += 1

    return patches, indices, stride_h, stride_w


def save_patches(image_folder, patch_folder, prefix):
    """
    Extracts patches from images in image_folder and saves them in patch_folder.

    Args:
        image_folder (str): Path to the folder containing images.
        patch_folder (str): Path to save patches.
        prefix (str): Prefix for naming output patches.
    """
    os.makedirs(patch_folder, exist_ok=True)  # Ensure output folder exists
    transform = transforms.ToTensor()
    image_counter = 0

    # Get sorted list of PNG files
    img_files = sorted([f for f in os.listdir(image_folder) if (f.endswith('.png') or f.endswith('.PNG'))])

    for img_file in img_files:
        img_path = os.path.join(image_folder, img_file)

        # Load and preprocess image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image_tensor = transform(img)  # Convert to tensor (C, H, W)

        # Extract patches
        patches, _, _, _ = extract_patches(image_tensor, patch_size=256)

        # Save patches efficiently
        for idx, patch in enumerate(patches):
            patch_name = f"{prefix}_{image_counter:05d}_{idx:04d}.png"
            patch_path = os.path.join(patch_folder, patch_name)
            transforms.ToPILImage()(patch).save(patch_path)

        image_counter += 1

        # Print progress every 100 images
        if image_counter % 10 == 0:
            print(f"Processed {image_counter} {prefix} images...")

    print(f"Finished processing {image_counter} {prefix} images.")


def extract_images_from_lmdb(lmdb_folder, output_folder):
    """
    Extracts PNG images from an LMDB database and saves them to the specified folder.

    Args:
        lmdb_folder (str): Path to the LMDB folder containing data.mdb and lock.mdb.
        output_folder (str): Path to the folder where extracted images will be saved.
    """
    # Open LMDB environment
    env = lmdb.open(lmdb_folder, readonly=True, lock=False)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            # Decode the key (filename)
            filename = key.decode("utf-8")

            # Ensure filename has .png extension
            if not filename.endswith(".png"):
                filename += ".png"

            # Convert value (image data) to numpy array
            img_array = np.frombuffer(value, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # Read as an image

            if img is None:
                print(f"Warning: Could not decode image for key {filename}, skipping...")
                continue  # Skip if decoding fails

            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # Save the image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)
            print(f"Saved {output_path}")

    print("Extraction complete!")