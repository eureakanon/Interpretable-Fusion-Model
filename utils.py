import torch
import torch.nn.functional as F
import numpy as np

def gaussian_blur_2d(img, kernel_size=11, sigma=3.0):
    """
    Apply Gaussian blur to a batch of images.
    Args:
        img: (B, C, H, W) normalized image tensor.
        kernel_size: int, odd.
        sigma: float.
    Returns:
        blurred: (B, C, H, W)
    """
    # Generate Gaussian kernel
    x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss.view(1, 1, -1, 1) @ gauss.view(1, 1, 1, -1)
    kernel = kernel.expand(img.size(1), 1, kernel_size, kernel_size)
    # Pad to maintain size
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding, groups=img.size(1))
    return blurred

def apply_perturbation(image, heatmap, blur_kernel_size=11, blur_sigma=3.0, threshold=0.5,temp=10.0):
    """
    Perturb the image by blurring regions where heatmap > threshold.
    Args:
        image: (B, 3, H, W) input image.
        heatmap: (B, 1, H, W) normalized heatmap in [0,1].
        blur_kernel_size: int.
        blur_sigma: float.
        threshold: float, regions above this are blurred.
    Returns:
        perturbed: (B, 3, H, W) perturbed image.
    """
    # Generate binary mask (1 for high activation, 0 otherwise)
    mask = torch.sigmoid((heatmap - threshold) * temp) 
    # Blur the whole image
    blurred = gaussian_blur_2d(image, kernel_size=blur_kernel_size, sigma=blur_sigma)
    # Combine: keep original where mask=0, use blurred where mask=1
    perturbed = image * (1 - mask) + blurred * mask
    return perturbed