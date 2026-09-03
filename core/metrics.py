"""
Evaluation Metrics & Anti-Hallucination Guardrails for Remote Sensing SRM.
Implements PSNR, SSIM, SAM (Spectral Angle Mapper), ERGAS,
Cycle-Consistency (Sensor PSF Degradation Error), and mIoU.
"""

import numpy as np


def calculate_psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) across multi-band imagery."""
    # Simplified multi-channel SSIM
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    mu_p = np.mean(pred)
    mu_g = np.mean(gt)
    sigma_p = np.var(pred)
    sigma_g = np.var(gt)
    sigma_pg = np.mean((pred - mu_p) * (gt - mu_g))
    
    ssim = ((2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)) / \
           ((mu_p ** 2 + mu_g ** 2 + c1) * (sigma_p + sigma_g + c2))
    return float(np.clip(ssim, 0.0, 1.0))


def calculate_sam(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculate Spectral Angle Mapper (SAM) in degrees.
    Critical remote sensing metric: measures spectral vector distortion.
    Lower is better (ideal = 0 degrees).
    """
    # Reshape to (N_pixels, C_channels)
    pred_vec = pred.reshape(-1, pred.shape[-1])
    gt_vec = gt.reshape(-1, gt.shape[-1])
    
    dot_product = np.sum(pred_vec * gt_vec, axis=1)
    norm_pred = np.linalg.norm(pred_vec, axis=1) + eps
    norm_gt = np.linalg.norm(gt_vec, axis=1) + eps
    
    cos_angle = np.clip(dot_product / (norm_pred * norm_gt), -1.0, 1.0)
    angles_deg = np.arccos(cos_angle) * (180.0 / np.pi)
    return float(np.mean(angles_deg))


def calculate_ergas(pred: np.ndarray, gt: np.ndarray, scale: int = 4) -> float:
    """
    Calculate Relative Dimensionless Global Error in Synthesis (ERGAS).
    Standard index for pan-sharpening and super-resolution in earth observation.
    Lower is better (typically < 3.0).
    """
    channels = pred.shape[-1]
    sum_rmse_ratio = 0.0
    for c in range(channels):
        p_c = pred[..., c]
        g_c = gt[..., c]
        rmse = np.sqrt(np.mean((p_c - g_c) ** 2))
        mean_g = np.mean(g_c) + 1e-8
        sum_rmse_ratio += (rmse / mean_g) ** 2
        
    ergas = (100.0 / scale) * np.sqrt((1.0 / channels) * sum_rmse_ratio)
    return float(ergas)


def calculate_cycle_consistency(sr_img: np.ndarray, lr_img: np.ndarray, scale: int = 4) -> float:
    """
    Anti-Hallucination Guardrail:
    Downsamples the 4x super-resolved image back to original resolution
    using sensor point-spread box averaging. Computes the absolute reconstruction error.
    Values close to 0 confirm no hallucinated features violate the input sensor data.
    """
    h, w, c = lr_img.shape
    # Block-average 4x4 patches
    degraded = sr_img.reshape(h, scale, w, scale, c).mean(axis=(1, 3))
    error = np.mean(np.abs(degraded - lr_img))
    return float(error)


def calculate_srm_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 5) -> dict:
    """
    Calculate Overall Accuracy and Mean Intersection-over-Union (mIoU)
    for Sub-Pixel Mapping (SRM).
    """
    oa = float(np.mean(pred_mask == gt_mask) * 100.0)
    
    ious = []
    for cls_idx in range(num_classes):
        intersection = np.sum((pred_mask == cls_idx) & (gt_mask == cls_idx))
        union = np.sum((pred_mask == cls_idx) | (gt_mask == cls_idx))
        if union == 0:
            continue
        ious.append(intersection / union)
        
    miou = float(np.mean(ious) * 100.0) if ious else 0.0
    return {"overall_accuracy": round(oa, 2), "miou": round(miou, 2)}
