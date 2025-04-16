# Reproducibility Challenge: Simple Baselines for Image Restoration
This repository contains our work for **York University’s EECS 6322: Neural Networks and Deep Learning Reproducibility Challenge**. We focus on the ECCV 2022 paper:

> **Simple Baselines for Image Restoration**  
> [Paper link (ECCV 2022)](https://arxiv.org/abs/2204.04676)

---

## ✅ Project Summary
With recent developments in Deep Learning and Computer Vision, **Image restoration** has become a trending topic in the field. However, most of the proposed methods for image restoration tasks deploy complex architectures as their base models. The paper we selected proposes a simple yet computationally efficient baseline that outperforms several similar, more complex models. Additionally, it shows that common non-linear activation functions—such as Sigmoid, ReLU, and Softmax—can be replaced by simple multiplication operations or even removed, introducing a **Nonlinear Activation-Free Network (NAFNet)**. Following the original paper's methodology, we reproduced the models and trained them on two public datasets: GoPro (for image deblurring) and SIDD (for image denoising), achieving peak signal-to-noise ratios (PSNR) of **30.83 dB** and **39.32 dB**, respectively.


---

## ✅ Our Contribution

We reproduced the original paper's models and training pipelines using **PyTorch**, and evaluated them on two public datasets:

## GoPro Dataset

| Model  | Baseline (Paper) | NAFNet (Paper) | Baseline (Ours)  | NAFNet (Ours)  |
|--------|------------------|----------------|------------------|----------------|
| PSNR   | 33.40            | 33.69          | 30.29            | 30.83          |
| SSIM   | 0.965            | 0.967          | 0.89             | 0.90           |


## SIDD Dataset

| Model  | Baseline (Paper) | NAFNet (Paper) | Baseline (Ours)  | NAFNet (Ours)  |
|--------|------------------|----------------|------------------|----------------|
| PSNR   | 40.30            | 40.30          | 39.24            | 39.32          |
| SSIM   | 0.962            | 0.962          | 0.92             | 0.92           |
---

## ✅ Project Structure

```bash
.
├── models/                # BaselineModel, NAFNetModel
├── notebooks/             # Jupyter notebooks for exploration
├── utils/                 # Helper functions: metrics, dataset loader, patcher
├── weights/               # Pretrained model weights
├── gopro_preprocess.py    # Extract & patch GoPro dataset
├── sidd_preprocess.py     # Extract & patch SIDD dataset
├── train.py               # Main training script 
├── test.py                # Main testing script
├── requirements.txt
└── README.md
```
---

 ## ✅ Instructions
 Make sure you have the following versions installed:
 - Python 3.11.0
 - CUDA 12.6 drivers (for GPU acceleration)

**1. Dependencies:**
   
 ```bash
 pip install -r requirements.txt
 ```
 This project uses PyTorch with GPU (CUDA 12.6). If you want to use the GPU version, install it using the following command:
 ```bash
 pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
 ```

**2. Data Preprocessing:**
   - Download the [**GoPro Train Set**](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view) and place it in the following directory:
  ```./datasets/GoPro/train/```
   - Download the [**GoPro Test Set**](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view), then move ```input.lmdb``` and ```target.lmdb``` files to the following directory:
  ```./datasets/GoPro/test/```
   - Run the following command:
     ```bash
     python gopro_preprocess.py
     ```

   - Download the [**SIDD Train Set**](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view) and place it in the following directory:
  ```./datasets/SIDD/train/```
   - Download the [**SIDD Test Set**](https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view), then move ```input_crops.lmdb``` and ```gt_crops.lmdb``` files to the following directory:
  ```./datasets/SIDD/test/```
   - Run the following command:
     ```bash
     python sidd_preprocess.py
     ```

 **3. Training:**
 - To train the models on each dataset, use the following commands:
     - Train **Baseline** model on **GoPro**:
     ```bash
     python train.py --model Baseline --dataset GoPro
     ```

     - Train **NAFNet** model on **GoPro**:
     ```bash
     python train.py --model NAFNet --dataset GoPro
     ```

     - Train **Baseline** model on **SIDD**:
     ```bash
     python train.py --model Baseline --dataset SIDD
     ```
     
     - Train **NAFNet** model on **SIDD**:
     ```bash
     python train.py --model NAFNet --dataset SIDD
     ```

  **4. Testing:**
  - To test the models on each dataset, use the following commands:
     - Test **Baseline** model on **GoPro**:
     ```bash
     python test.py --model Baseline --dataset GoPro
     ```

     - Test **NAFNet** model on **GoPro**:
     ```bash
     python test.py --model NAFNet --dataset GoPro
     ```

     - Test **Baseline** model on **SIDD**:
     ```bash
     python test.py --model Baseline --dataset SIDD
     ```
     
     - Test **NAFNet** model on **SIDD**:
     ```bash
     python test.py --model NAFNet --dataset SIDD
     ```

