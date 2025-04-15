import torch
import torch.nn.functional as F

def psnr_loss(pred, target, max_val=1.0):
    """
    Compute PSNR loss
    Args:
        pred (torch.Tensor): Predicted image tensor (B, C, H, W)
        target (torch.Tensor): Target image tensor (B, C, H, W)
        max_val (float): Maximum pixel value
    """
    mse = F.mse_loss(pred, target, reduction='none')  # Per-pixel MSE
    mse = mse.reshape(mse.shape[0], -1).mean(dim=1)  # Mean over spatial dimensions, per image
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-9))  # PSNR per image
    return -psnr.mean()


def ssim_loss(pred, target, window_size=11, max_val=1.0):
    """
    Compute SSIM loss
    Args:
        pred (torch.Tensor): Predicted image tensor (B, C, H, W)
        target (torch.Tensor): Target image tensor (B, C, H, W)
        window_size (int): Size of the Gaussian window
        max_val (float): Maximum pixel value
    """
    C1 = (0.01 * max_val) ** 2  # Constant for luminance
    C2 = (0.03 * max_val) ** 2  # Constant for contrast

    # Define Gaussian kernel
    def gaussian_kernel(window_size, sigma=1.5):
        kernel = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
        kernel /= kernel.sum()
        return kernel[:, None] * kernel[None, :]

    kernel = gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)  # (C, 1, W, W)

    # Compute means
    mu_pred = F.conv2d(pred, kernel, groups=pred.shape[1], padding=window_size // 2)
    mu_target = F.conv2d(target, kernel, groups=target.shape[1], padding=window_size // 2)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    # Compute variances
    sigma_pred_sq = F.conv2d(pred * pred, kernel, groups=pred.shape[1], padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, kernel, groups=target.shape[1], padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, groups=pred.shape[1], padding=window_size // 2) - mu_pred_target

    # Compute SSIM per pixel
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
                (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim.mean()
