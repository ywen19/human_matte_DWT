"""
Common metrics for matting tasks;
the following definitions strictly follow the metrics computation
and formulations from MaGGIe (2024):
https://github.com/hmchuong/MaGGIe/blob/main/maggie/utils/metric.py
"""

import os
import sys

import math
import numpy as np

import torch
import torch.nn.functional as F

import cv2

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.utils import _unknown_mask


_GD_CACHE = {}  # for GRAD


class AvgMeter:
    """
    current computation mode only works when reduction='none' in metrics
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.avg:.4f} ({self.sum:.1f}/{self.count})"


@torch.no_grad()
def compute_mad_full(alpha_pred: torch.Tensor,
                     alpha_gt: torch.Tensor) -> torch.Tensor:
    # per-image mean(|err|) * 1000  -> (B,)
    return (alpha_pred - alpha_gt).abs().flatten(1).mean(dim=1) * 1000.0


@torch.no_grad()
def compute_sad(
    alpha_pred: torch.Tensor,
    alpha_gt:   torch.Tensor,
    trimap:     torch.Tensor,  # SAD is calculated in unknown refion only
    reduction:  str = 'mean',  # aggregation of per-image SAD：'none' | 'mean' | 'sum'
    scaled:     bool = True,   # True -> /1000 as a general scale in matting research
) -> torch.Tensor:
    diff = (alpha_pred - alpha_gt).abs()                      # (B,1,H,W)
    mask = _unknown_mask(trimap).to(diff.dtype)               # (B,1,H,W)

    sad_per_img = (diff * mask).flatten(1).sum(dim=1)         # (B,)
    if scaled:
        sad_per_img = sad_per_img / 1000.0

    if reduction == 'none':
        return sad_per_img
    elif reduction == 'mean':
        return sad_per_img.mean()
    elif reduction == 'sum':
        return sad_per_img.sum()
    else:
        raise ValueError("reduction must be 'none' | 'mean' | 'sum'")


@torch.no_grad()
def compute_mse_unknown(
    alpha_pred: torch.Tensor,
    alpha_gt:   torch.Tensor,
    trimap:     torch.Tensor,
    reduction:  str  = 'none',   # 'none'，for whole dataset: sum/n 聚合
    scaled:     bool = True,     # True -> ×1e3 as general scale
) -> torch.Tensor:
    mask = _unknown_mask(trimap).to(alpha_pred.dtype)                     # (B,1,H,W)
    num = ((alpha_pred - alpha_gt) ** 2 * mask).flatten(1).sum(dim=1)    # (B,)
    den = mask.flatten(1).sum(dim=1) + 1e-8                              # (B,)
    mse = num / den                                                      # (B,)
    if scaled:
        mse = mse * 1e3
    if reduction == 'none':
        return mse
    if reduction == 'mean':
        return mse.mean()
    if reduction == 'sum':
        return mse.sum()
    raise ValueError("reduction must be 'none'|'mean'|'sum'")


def _gauss_1d(x, sigma):
    return torch.exp(-x * x / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))


def _dgauss_1d(x, sigma):
    g = _gauss_1d(x, sigma)
    return -x * g / (sigma * sigma)


def _get_gauss_deriv_kernels(sigma: float, device, dtype, eps: float = 1e-2):
    key = (float(sigma), str(device), dtype)
    if key in _GD_CACHE:
        return _GD_CACHE[key]

    half_size = math.ceil(sigma * math.sqrt(-2.0 * math.log(math.sqrt(2.0 * math.pi) * sigma * eps)))
    size = int(2 * half_size + 1)

    xs = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
    g = _gauss_1d(xs, sigma)      # (size,)
    dg = _dgauss_1d(xs, sigma)     # (size,)

    # 2D filter kernel：filter_x(y,x) = G(y) * dG(x)；filter_y = filter_x^T
    filter_x = torch.outer(g, dg)  # (size,size)
    norm = torch.sqrt((filter_x * filter_x).sum()).clamp_min(1e-12)
    filter_x = filter_x / norm
    filter_y = filter_x.t()

    fx = filter_x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    fy = filter_y.unsqueeze(0).unsqueeze(0)
    _GD_CACHE[key] = (fx, fy)
    return fx, fy


@torch.no_grad()
def compute_gradient_error_maggie(
    alpha_pred: torch.Tensor,   # (B,1,H,W), in [0,1]
    alpha_gt:   torch.Tensor,   # (B,1,H,W), in [0,1]
    trimap:     torch.Tensor,   # (B,1,H,W), 0/128/255
    sigma:      float = 1.4,
    reduction:  str = 'mean',  # 'none' | 'mean' | 'sum'  (aggregation of per-image sum)
) -> torch.Tensor:
    """
    MaGGIe-style Grad (Unknown region):
      1) Per-image min-max normalization to [0,1]
      2) Gaussian first-order gradient (sigma=1.4), magnitude = sqrt(gx^2 + gy^2)
      3) Compute pixel-wise sum of (|∇pred| - |∇gt|)^2 inside the Unknown region
         (no averaging, no scaling)
      4) Aggregate by reduction (typically: use 'none' to get per-image vector,
         then AvgMeter for equal-weight averaging)
    """
    device, dtype = alpha_pred.device, alpha_pred.dtype
    fx, fy = _get_gauss_deriv_kernels(sigma, device, dtype)
    pad = fx.shape[-1] // 2

    def _minmax01(x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    pred_n = _minmax01(alpha_pred)
    gt_n = _minmax01(alpha_gt)

    # gradient magnitude from Gaussian derivative
    gx_p = F.conv2d(pred_n, fx, padding=pad)
    gy_p = F.conv2d(pred_n, fy, padding=pad)
    grad_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p)

    gx_g = F.conv2d(gt_n, fx, padding=pad)
    gy_g = F.conv2d(gt_n, fy, padding=pad)
    grad_g = torch.sqrt(gx_g * gx_g + gy_g * gy_g)

    # Unknown mask（B,1,H,W）
    mask = _unknown_mask(trimap).to(dtype=dtype)

    # per-image pixel sum (no normalization, no scaling)
    diff = (grad_p - grad_g) ** 2
    per_img = (diff * mask).flatten(1).sum(dim=1)  # (B,)

    if reduction == 'none':
        return per_img
    elif reduction == 'mean':
        return per_img.mean()
    elif reduction == 'sum':
        return per_img.sum()
    else:
        raise ValueError("reduction must be 'none' | 'mean' | 'sum'")


@torch.no_grad()
def compute_connectivity_error_maggie(
    alpha_pred: torch.Tensor,
    alpha_gt:   torch.Tensor,
    trimap:     torch.Tensor,
    step:       float = 0.1,
    reduction:  str = 'mean',
    scaled:     bool = True,  # True -> ×0.001
) -> torch.Tensor:
    """
    MaGGIe-style Conn on Unknown:
      Per-image:
        - For a sequence of thresholds, compute LCC on GT∩Pred to build round_down_map;
        - φ(x) = 1 - (x - round_down), applied only if the difference ≥ 0.15;
        - On the Unknown region, compute pixel sum of |φ_gt - φ_pred|;
        - Finally multiply by 0.001 (scaled=True).
      Aggregation:
        - Aggregate per-image sums with equal weighting (controlled by reduction).
    """
    B, _, H, W = alpha_pred.shape
    pred_np = alpha_pred.squeeze(1).detach().cpu().numpy()  # (B,H,W)
    gt_np = alpha_gt  .squeeze(1).detach().cpu().numpy()  # (B,H,W)
    trimap_np = trimap    .squeeze(1).detach().cpu().numpy()  # (B,H,W)

    thresholds = np.arange(0.0, 1.0 + step, step, dtype=np.float32)
    errs = []

    for b in range(B):
        pred_b = pred_np[b]
        gt_b = gt_np[b]

        # Unknown ROI
        roi_mask = _unknown_mask(torch.from_numpy(trimap_np[b])).numpy().astype(np.float32)

        # round_down_map
        rdm = -np.ones_like(gt_b, dtype=np.float32)

        for i in range(1, len(thresholds)):
            t = thresholds[i]
            t_prev = thresholds[i - 1]

            gt_bin = (gt_b >= t)
            pred_bin = (pred_b >= t)
            inter = (gt_bin & pred_bin).astype(np.uint8)

            if inter.any():
                # LCC of intersection, 4-connectivity (equivalent to skimage connectivity=1)
                num, labels = cv2.connectedComponents(inter, connectivity=4)
                if num > 1:
                    counts = np.bincount(labels.ravel())[1:]  # skip background
                    max_id = counts.argmax() + 1
                    omega = (labels == max_id).astype(np.uint8)
                else:
                    omega = np.zeros_like(inter, dtype=np.uint8)
            else:
                omega = np.zeros_like(inter, dtype=np.uint8)

            # first dropout → take previous threshold
            drop_mask = (rdm == -1) & (omega == 0)
            rdm[drop_mask] = t_prev

        # if pixel remains -1 (never dropped), assign 1
        rdm[rdm == -1] = 1.0

        # φ mapping (only applied when difference >= 0.15)
        gt_diff = gt_b - rdm
        pred_diff = pred_b - rdm
        gt_phi = 1.0 - gt_diff * (gt_dif >= 0.15)
        pred_phi = 1.0 - pred_diff * (pred_diff >= 0.15)
        # pixel sum for a single image
        conn_sum = np.sum(np.abs(gt_phi - pred_phi) * roi_mask).astype(np.float32)
        errs.append(conn_sum)

    errors = torch.tensor(errs, dtype=torch.float32)  # (B,)
    if scaled:
        errors = errors * 1e-3  # align with MaGGIe's compute_metric by multiplying 0.001

    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        return errors.mean()
    elif reduction == 'sum':
        return errors.sum()
    else:
        raise ValueError("reduction must be 'none' | 'mean' | 'sum'")