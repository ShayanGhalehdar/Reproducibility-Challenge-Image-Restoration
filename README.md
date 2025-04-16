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

## (GoPro Dataset)

| Method | Baseline (Paper) | NAFNet (Paper) | Baseline (Ours) | NAFNet (Ours) |
|--------|------------------|----------------|------------------|----------------|
| PSNR   | 33.40            | 33.69          | 30.29            | 30.83          |
| SSIM   | 0.965            | 0.967          | 0.89             | 0.90           |

---

## ✅ Project Structure

```bash
.
├── models/                # BaselineModel, NAFNetModel
├── utils/                 # Loss functions, dataset loader, patching tools
├── GoPro/                 # Preprocessed GoPro dataset
├── SIDD/                  # Preprocessed SIDD dataset
├── train.py               # Main training script (model + dataset args)
├── gopro_preprocess.py    # Extract & patch GoPro data
├── sidd_preprocess.py     # Extract & patch SIDD data
├── requirements.txt
└── README.md
