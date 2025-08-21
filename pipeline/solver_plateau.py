import os
import sys
import glob
import re
import gc
import itertools
from typing import List, Tuple, Optional
import json 

import math
import numpy as np
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torchvision.utils as vutils
from pytorch_wavelets import DWTForward

import cv2
import kornia
from kornia.losses import SSIMLoss
from pytorch_wavelets import DWTForward           
from torchvision.transforms.functional import gaussian_blur

# 开启异常检测，以便在反向传播时立即定位第一个 NaN/Inf 的来源
torch.autograd.set_detect_anomaly(True)

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
# from model.refiner_xnet_rgbembed import refiner_xnet
from model.refiner_xnet_noattn import refiner_xnet

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ================= Sobel / Edge-Weight 相关超参 =================
WARM_EPOCHS        = 2         # Sobel 权重 warm‑up 轮数
TARGET_FRAC        = 0.15      # Sobel_raw 占主干损失期望比例
MAX_SOBEL_GAIN     = 2.0       # Sobel 整体最大放大倍数
GAIN_RAMP_EPOCHS   = 10        # gain 从 1→MAX 的线性 epoch 数

EDGE_K_START       = 1.0       # 边缘像素权重初值 k₀
EDGE_K_END         = 3.0       # 边缘像素权重最终值 k₁
EDGE_K_RAMP_EPOCHS = 6         # k 从 k₀→k₁ 的余弦上升 epoch 数
EDGE_K_MAX_CLIP    = 5.0       # （可选）对 edge_k·|∇α| 的 clip 上限


cfg = {
    'num_epochs':       21,
    'batch_size':       2,
    "accum_step":       4,
    'checkpoint_dir':   'checkpoints_xnet_db1_final_noattn',
    'log_dir':          'log_xnet_db1_final',
    'csv_path':         '../data/pair_for_refiner.csv',
    'vis_dir':          'vis_xnet_db1_final_test',
    'vis_val_dir':      'vis_val_xnet_db1_noattn_test',
    'viz_interval':     64,
    'lr':               1e-4,
    'weight_decay':     1e-5,
    'seed':             3407,
    'wavelet_list':     ['db1', 'db1'],
    'd_head':           128,
    'base_channel':     64
}

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
if device_type == "cuda":
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)

def pick_gn_groups(C: int) -> int:
    # 优先 32/16/8/4/2/1，确保能整除通道数
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1

def bn_to_gn(module: nn.Module):
    for name, m in list(module.named_children()):
        if isinstance(m, nn.BatchNorm2d):
            C = m.num_features
            gn = nn.GroupNorm(num_groups=pick_gn_groups(C), num_channels=C, affine=True, eps=1e-5)
            setattr(module, name, gn)
        else:
            bn_to_gn(m)

def count_bn(module):
    return sum(isinstance(m, nn.BatchNorm2d) for m in module.modules())

def count_gn(module):
    return sum(isinstance(m, nn.GroupNorm) for m in module.modules())

def _unknown_mask(trimap: torch.Tensor) -> torch.Tensor:
    """
    Return float mask ==1 on Unknown region, same shape as trimap.

    Accepts
    -------
    uint8  : {0,128,255}
    float  : {0,0.5,1}  or  {0,128,255}
    """
    # → float32, 0–1
    if not trimap.dtype.is_floating_point:
        t = trimap.float() / 255.0
    else:
        t = trimap.clone()
        if t.max() > 1.0:
            t = t / 255.0
    return ((t > 0.01) & (t < 0.99)).float()  # Unknown 区域为 1

def log_gpu_usage(tag: str = "") -> None:
    """Print per‑GPU memory stats (allocated/reserved vs whole‑card used)."""
    for idx in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(idx) / 1024 ** 2
        reserv = torch.cuda.memory_reserved(idx) / 1024 ** 2
        free, total = torch.cuda.mem_get_info(idx)
        used = (total - free) / 1024 ** 2
        total /= 1024 ** 2
        print(
            f"[{tag}] GPU{idx}: allocated={alloc:.1f} MB, "
            f"reserved={reserv:.1f} MB, used={used:.1f}/{total:.1f} MB (whole card)",
            flush=True,
        )


def loss_l1(alpha_pred: torch.Tensor,
            alpha_gt:   torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        diff = (alpha_pred - alpha_gt).abs() * mask
        loss = diff.sum() / (mask.sum() + 1e-8)   # 避免除 0
    else:
        loss = F.l1_loss(alpha_pred, alpha_gt)
    return loss

def get_edge_k(epoch: int) -> float:
    """
    返回当前 epoch 使用的 edge_k（乘到 |∇α_gt| 上的放大系数）

    0–EDGE_K_RAMP_EPOCHS 采用半余弦曲线 k₀→k₁，之后恒 k₁。
    """
    if epoch >= EDGE_K_RAMP_EPOCHS:
        return EDGE_K_END
    ratio = 0.5 * (1.0 - math.cos(math.pi * epoch / EDGE_K_RAMP_EPOCHS))
    return EDGE_K_START + (EDGE_K_END - EDGE_K_START) * ratio

def compute_sobel_weights(l1_full_val: float,
                          sobel_full_val: float,
                          sobel_u_val: float,
                          grad_val: float,
                          epoch: int) -> tuple[float, float]:
    """输出 (w_sf, w_su)，分别乘在 sobel_full / sobel_u_edge 上"""
    # 1) warm‑up：固定 1
    if epoch < WARM_EPOCHS:
        return 1.0, 1.0

    # 2) 当前三项 raw‑loss
    sobel_raw = sobel_full_val + sobel_u_val
    main_raw  = l1_full_val + sobel_raw + grad_val

    # 3) 让 sobel_raw ≈ TARGET_FRAC × main_raw
    scale = TARGET_FRAC * main_raw / max(sobel_raw, 1e-8)

    # 4) epoch‑level gain：1 → MAX_SOBEL_GAIN
    gain = 1.0 + (MAX_SOBEL_GAIN - 1.0) * \
           min(1.0, (epoch - WARM_EPOCHS + 1) / GAIN_RAMP_EPOCHS)

    scale *= gain
    scale = max(0.5, min(scale, MAX_SOBEL_GAIN))   # 软裁剪

    return scale, scale

def multiscale_sobel_l1(
        pred:  torch.Tensor,        # (B,1,H,W) α∈[0,1]
        gt:    torch.Tensor,        # (B,1,H,W)
        scales                   = (1., 0.5, 0.25),
        region: str               = "full",   # "full" | "unknown" | "mask"
        trimap: torch.Tensor | None = None,   # 需要用于 region="unknown"
        custom_mask: torch.Tensor | None = None,
        edge_weighted: bool       = False,
        edge_k: float             = 3.0,      # 权重 = k*|∇α_gt|
        edge_kmax: float          = 4.0,      # clip 上限
        reduction: str            = "mean",
        # --- 新增稳健性参数 ---
        min_mask_px: int          = 1024,     # 掩码像素过少则退化全图
        min_mask_ratio: float     = 0.001,    # 占比阈值（与上面取 max）
) -> torch.Tensor:
    """
    多尺度 Sobel-L1，支持区域 (full/unknown/mask)、可选 edge weighting。
    若区域掩码像素过少，则自动退化为全图平均，避免分母~0 导致爆梯度。
    """

    if not isinstance(scales, (list, tuple)):
        scales = (float(scales),)

    # ---------- 构造初始 mask ----------
    if region == "full":
        mask = None
    elif region == "unknown":
        assert trimap is not None, "`trimap` required when region='unknown'"
        # _unknown_mask: 输入 trimap∈{0,128,255} -> (B,1,H,W) float{0,1}
        mask = _unknown_mask(trimap)  # 你文件中已有实现
    elif region == "mask":
        assert custom_mask is not None, "`custom_mask` required when region='mask'"
        mask = custom_mask.float()
    else:
        raise ValueError(f"region must be 'full|unknown|mask', got {region}")

    loss_acc = 0.
    for s in scales:
        # --- 下采样到当前尺度 ---
        if s != 1.0:
            size = (int(gt.shape[-2] * s), int(gt.shape[-1] * s))
            p_ds = F.interpolate(pred,  size, mode="bilinear", align_corners=False)
            g_ds = F.interpolate(gt,    size, mode="bilinear", align_corners=False)
            m_ds = None if mask is None else F.interpolate(mask, size, mode="nearest")
        else:
            p_ds, g_ds, m_ds = pred, gt, mask

        # Sobel 梯度 (dx, dy)
        grad_p = kornia.filters.sobel(p_ds)  # (B,2,H,W)
        grad_g = kornia.filters.sobel(g_ds)  # (B,2,H,W)
        diff   = (grad_p - grad_g).abs()     # (B,2,H,W)

        # ---------- Edge weighting ----------
        if edge_weighted:
            edge_w = kornia.filters.sobel(g_ds).abs().sum(1, keepdim=True)  # (B,1,H,W)
            edge_w = torch.clip(edge_w * edge_k + 1.0, 1.0, edge_kmax)
            diff   = diff * edge_w  # broadcast 至 2 通道

        # ---------- 区域 Mask + 最小像素数保护 ----------
        use_mask = False
        if m_ds is not None:
            # m_ds: (B,1,H,W) ∈{0,1}
            mask_px = m_ds.sum()  # 所有 batch 合计
            if min_mask_ratio is not None:
                total_px = torch.tensor(float(m_ds.numel() // m_ds.shape[1]), device=m_ds.device)
                # m_ds.numel() = B*1*H*W; //m_ds.shape[1]去掉通道1 -> B*H*W
                mask_thresh = max(float(min_mask_px), float(min_mask_ratio) * total_px.item())
            else:
                mask_thresh = float(min_mask_px)
            if mask_px >= mask_thresh:
                use_mask = True

        if use_mask:
            diff = diff * m_ds  # 广播到 2 通道
            # denom: Unknown像素 * 2（dx,dy）
            denom = mask_px * 2.0
        else:
            # 掩码太小：退化全图
            denom = diff.numel()

        # ---------- reduction ----------
        if reduction == "mean":
            loss_acc += diff.sum() / (denom + 1e-8)
        elif reduction == "sum":
            # 注意：sum 时不做退化缩放（保持原行为）；如果你想保持一致，可改 denom
            loss_acc += diff.sum() if use_mask else diff.sum()
        elif reduction == "none":
            # 返回逐元素；若 use_mask=True，已零掉非掩码元素
            loss_acc += diff
        else:
            raise ValueError("reduction must be mean|sum|none")

    return loss_acc / len(scales)

@torch.no_grad()
def conv_dwt_energy_mask(
    alpha_gt: torch.Tensor,
    kernels: torch.Tensor,  # (3,1,2,2) - Haar 高通核
    threshold_quantile: float = 0.90,
    dilation_radius: int = 2,
) -> torch.Tensor:
    """
    基于 conv 的 DWT 高频能量计算 edge band 掩码。
    1) 使用 Haar 高通核计算 HF；
    2) 对 HF 求能量（sqrt(sum of squares)）；
    3) 对能量按 quantile 阈值二值化；
    4) 膨胀为 band 掩码。

    Returns:
        band: (B, 1, H, W) float{0,1}
    """
    alpha = alpha_gt.float()
    hp = F.conv2d(alpha, kernels.to(alpha.device), padding=1)  # (B,3,H,W)
    energy = torch.sqrt((hp ** 2).sum(dim=1, keepdim=False) + 1e-6)  # (B,H,W)
    energy = energy.unsqueeze(1)  # (B,1,H,W)

    # 按 batch 求 quantile
    B = alpha.size(0)
    band_masks = []
    for b in range(B):
        e_b = energy[b]
        th = torch.quantile(e_b.view(-1), threshold_quantile)
        mask = (e_b > th).float()
        # 膨胀
        k = 2 * dilation_radius + 1
        band = F.max_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=dilation_radius)
        band_masks.append(band)

    return torch.cat(band_masks, dim=0)  # (B,1,H,W)


def _sobel_edge_weight(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradient magnitude normalized to [0,1] for alpha maps.
    """
    # Create Sobel kernels with same dtype as alpha
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      device=alpha.device,
                      dtype=alpha.dtype).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx = F.conv2d(alpha, kx, padding=1)
    gy = F.conv2d(alpha, ky, padding=1)
    grad = torch.sqrt(gx*gx + gy*gy)
    maxv = grad.amax(dim=[2,3], keepdim=True).clamp(min=1e-6)
    return (grad / maxv).clamp(0,1)

class DWTLoss(nn.Module):
    def __init__(
        self,
        hf_weighted: bool = True,
        enable_band: bool = True,
        band_radius: int = 2,
        band_weight: float = 2.0,
        eps: float = 1e-6,
        threshold_quantile: float = 0.90,
    ):
        super().__init__()
        # Haar高通核 (模拟 DWT)
        kernels = torch.tensor([
            [[-0.5,  0.5], [-0.5,  0.5]],  # LH
            [[-0.5, -0.5], [ 0.5,  0.5]],  # HL
            [[ 0.5, -0.5], [-0.5,  0.5]],  # HH
        ], dtype=torch.float32)  # shape (3, 2, 2)
        self.register_buffer('kernels', kernels.unsqueeze(1))  # → (3,1,2,2)

        self.hf_weighted       = hf_weighted
        self.enable_band       = enable_band
        self.band_radius       = band_radius
        self.band_weight       = band_weight
        self.eps               = eps
        self.th_q              = threshold_quantile

    def forward(
        self,
        alpha_pred: torch.Tensor,  # (B,1,H,W), might be half precision
        alpha_gt:   torch.Tensor,  # (B,1,H,W)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # HF diff
        kernels = self.kernels.to(dtype=alpha_pred.dtype, device=alpha_pred.device)
        hp_pred = F.conv2d(alpha_pred, kernels, padding=1)
        hp_gt   = F.conv2d(alpha_gt,   kernels, padding=1)
        diff    = (hp_pred - hp_gt).abs()
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=1e6)

        # Sobel edge weighting
        if self.hf_weighted:
            w = _sobel_edge_weight(alpha_gt)
            if w.shape[-2:] != diff.shape[-2:]:
                w = F.interpolate(w, size=diff.shape[-2:], mode='bilinear', align_corners=False)
            diff = diff * w.expand_as(diff)
            diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=1e6)

        hf_full = diff.mean()
        hf_band = None

        if self.enable_band and self.band_radius >= 0:
            gt_f32 = alpha_gt.to(torch.float32).to(alpha_pred.device)
            band = conv_dwt_energy_mask(
                gt_f32, kernels, self.th_q, self.band_radius
            )
            if band.shape[-2:] != diff.shape[-2:]:
                band = F.interpolate(band, size=diff.shape[-2:], mode="nearest")
            band = band.to(dtype=diff.dtype, device=diff.device)
            diff_band = diff * torch.where(band > 0, self.band_weight, 1.0)
            diff_band = torch.nan_to_num(diff_band, nan=0.0, posinf=1e6, neginf=1e6)
            hf_band = diff_band.mean()

        return hf_full, hf_band


class ConnectivityLoss(nn.Module):
    """Matting connectivity loss.

    Parameters
    ----------
    unknown_only : bool, default True
        If True, loss is *averaged* only over trimap Unknown 区域
        (trimap==0.5 *or* 128).  If False,全图平均。
    step : float, default 0.01
        Threshold stride ∆T (0–1). 0.01 => 100 masks per forward.
    """
    def __init__(self, unknown_only: bool = True, step: float = 0.01):
        super().__init__()
        self.unknown_only = unknown_only
        thr = torch.arange(0.0, 1.0 + 1e-6, step)  # [0,1)
        self.register_buffer("thr", thr.view(1, -1, 1, 1))  # (1,K,1,1)

    @torch.no_grad()
    def _largest_cc(self, bi_map: torch.Tensor) -> torch.Tensor:
        """Keep only largest 4‑connected component (B,1,H,W)."""
        # dilation‑style flooding until convergence (32 iter for 4k² tops)
        comp = bi_map.clone()
        for _ in range(32):
            comp = F.max_pool2d(comp, 3, stride=1, padding=1)
        return comp

    def forward(self, alpha_pred: torch.Tensor, alpha_gt: torch.Tensor,
                trimap: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = alpha_pred.shape
        device = alpha_pred.device
        thr = self.thr.to(device)               # (1,K,1,1)
        K = thr.shape[1]

        # Binary masks > t for all thresholds at once -> (B,K,1,H,W)
        pred_bin = (alpha_pred.unsqueeze(1) > thr).float()
        gt_bin   = (alpha_gt  .unsqueeze(1) > thr).float()

        # Largest connected component of GT & pred for each threshold
        gt_lcc   = self._largest_cc(gt_bin.view(-1,1,H,W)).view(B, K, 1, H, W)
        pred_lcc = self._largest_cc(pred_bin.view(-1,1,H,W)).view(B, K, 1, H, W)

        # Pixel is correct if it belongs to GT‑LCC ∧ Pred‑LCC simultaneously
        match = (pred_bin * gt_lcc) * (gt_bin * pred_lcc)
        err_map = 1.0 - match.squeeze(2)        # (B,K,H,W)  1=error

        # Aggregate over thresholds (mean over K) → per‑pixel error (B,1,H,W)
        err_px = err_map.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # ==== Unknown 区域掩码 ====
        if self.unknown_only and trimap is not None:
            if trimap.dtype.is_floating_point:
                unk_mask = (trimap == 0.5).float()
            else:  # uint8 / int
                unk_mask = (trimap == 128).float()
            err_px = err_px * unk_mask
            denom = unk_mask.sum() + 1e-8
        else:
            denom = torch.numel(err_px)

        loss = err_px.sum() / denom
        return loss


def gradient_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask:   torch.Tensor | None = None,
        charbonnier: bool = True,
        eps: float = 1e-3,
        reduction: str = "mean",
        # --- 新增稳健性参数 ---
        min_mask_px: int = 1024,       # 掩码像素过少时退化全图
        min_mask_ratio: float = 0.001, # 掩码占比过小时退化全图
) -> torch.Tensor:
    """
    计算 Sobel‑based 的梯度损失，支持对 Unknown 区域加权。
    当掩码(mask)下的像素数过少时，自动退化为全图平均，避免除以近零。
    """

    # 1) 先算 Sobel 梯度
    grad_p = kornia.filters.sobel(pred)    # (B,2,H,W)
    grad_t = kornia.filters.sobel(target)  # (B,2,H,W)

    diff = grad_p - grad_t                  # (B,2,H,W)
    if charbonnier:
        diff = torch.sqrt(diff * diff + eps * eps)
    else:
        diff = diff.abs()

    # 2) 掩码模式
    if mask is not None:
        # 如果传进来的是 trimap (0/128/255) 而非布尔 mask
        if mask.dtype != torch.bool and (mask.max() > 1.0 or not torch.all((mask == 0) | (mask == 1))):
            mask = _unknown_mask(mask)      # 你文件中已有实现

        # 统计掩码像素数
        mask_px = mask.sum()               # B*H*W 合计
        total_px = mask.numel() // mask.shape[1]  # B*H*W（去掉通道1）
        thresh = max(float(min_mask_px), float(min_mask_ratio) * total_px)

        if mask_px < thresh:
            # 掩码像素太少，退化全图平均
            if reduction == "mean":
                return diff.mean()
            elif reduction == "sum":
                return diff.sum()
            elif reduction == "none":
                return diff
            else:
                raise ValueError(f"Invalid reduction: {reduction}")

        # 正常掩码模式：只在掩码区域计算
        diff = diff * mask               # broadcast 到 2 通道
        if reduction == "mean":
            # ×2 因为有 X、Y 两个梯度通道
            return diff.sum() / (mask_px * 2.0 + eps)
        elif reduction == "sum":
            return diff.sum()
        elif reduction == "none":
            return diff
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    # 3) 全图模式
    else:
        if reduction == "mean":
            return diff.mean()
        elif reduction == "sum":
            return diff.sum()
        elif reduction == "none":
            return diff
        else:
            raise ValueError(f"Invalid reduction: {reduction}")


################################### 评分指标 #####################################
class AvgMeter:
    """
    当前的计算模式只适用于当metrics计算时reduction模式是none
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
    trimap:     torch.Tensor,          # 只支持 unknown，必须提供
    reduction:  str = 'mean',          # 对“每图 SAD”的聚合：'none' | 'mean' | 'sum'
    scaled:     bool = True,           # True -> /1000 便于展示
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
    reduction:  str  = 'none',   # 推荐 'none'，后面用 sum/n 聚合
    scaled:     bool = True,     # True -> ×1e3 展示
) -> torch.Tensor:
    mask = _unknown_mask(trimap).to(alpha_pred.dtype)                     # (B,1,H,W)
    num  = ((alpha_pred - alpha_gt) ** 2 * mask).flatten(1).sum(dim=1)    # (B,)
    den  = mask.flatten(1).sum(dim=1) + 1e-8                              # (B,)
    mse  = num / den                                                      # (B,)
    if scaled:
        mse = mse * 1e3
    if reduction == 'none': return mse
    if reduction == 'mean': return mse.mean()
    if reduction == 'sum':  return mse.sum()
    raise ValueError("reduction must be 'none'|'mean'|'sum'")

_GD_CACHE = {}

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
    g  = _gauss_1d(xs, sigma)      # (size,)
    dg = _dgauss_1d(xs, sigma)     # (size,)

    # 2D 分离核：filter_x(y,x) = G(y) * dG(x)；filter_y = filter_x^T
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
    reduction:  str   = 'mean',  # 'none' | 'mean' | 'sum'  (对“每图和”的聚合)
) -> torch.Tensor:
    """
    MaGGIe-style Grad (Unknown 区域):
      1) 逐图 min-max 归一化到 [0,1]
      2) 高斯一阶导梯度 (sigma=1.4)，幅值 sqrt(gx^2+gy^2)
      3) 在 Unknown 内计算 (|∇pred|-|∇gt|)^2 的像素和（不平均、不缩放）
      4) 最后按 reduction 聚合（通常：先 'none' 得到逐图向量，再 AvgMeter 做等权平均）
    """
    device, dtype = alpha_pred.device, alpha_pred.dtype
    fx, fy = _get_gauss_deriv_kernels(sigma, device, dtype)
    pad = fx.shape[-1] // 2

    def _minmax01(x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    pred_n = _minmax01(alpha_pred)
    gt_n   = _minmax01(alpha_gt)

    # 高斯导数梯度幅值
    gx_p = F.conv2d(pred_n, fx, padding=pad)
    gy_p = F.conv2d(pred_n, fy, padding=pad)
    grad_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p)

    gx_g = F.conv2d(gt_n, fx, padding=pad)
    gy_g = F.conv2d(gt_n, fy, padding=pad)
    grad_g = torch.sqrt(gx_g * gx_g + gy_g * gy_g)

    # Unknown 区域 mask（B,1,H,W）
    mask = _unknown_mask(trimap).to(dtype=dtype)

    # 单图像素和（不做归一化、不做缩放）
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
    reduction:  str   = 'mean',
    scaled:     bool  = True,       # True -> ×0.001
) -> torch.Tensor:
    """
    MaGGIe-style Conn on Unknown:
      单图：按阈值序列在 GT∩Pred 上求 LCC，构造 round_down_map；
            φ(x)=1 - (x - round_down) 但仅当差值≥0.15；然后在 Unknown 上求 |φ_gt-φ_pred| 的像素和；
            最后 ×0.001（scaled=True）。
      汇总：对“每图和”做逐图等权聚合（由 reduction 控制）。
    """
    B, _, H, W = alpha_pred.shape
    pred_np   = alpha_pred.squeeze(1).detach().cpu().numpy()  # (B,H,W)
    gt_np     = alpha_gt  .squeeze(1).detach().cpu().numpy()  # (B,H,W)
    trimap_np = trimap    .squeeze(1).detach().cpu().numpy()  # (B,H,W)

    thresholds = np.arange(0.0, 1.0 + step, step, dtype=np.float32)
    errs = []

    for b in range(B):
        pred_b = pred_np[b]
        gt_b   = gt_np[b]

        # Unknown ROI
        roi_mask = _unknown_mask(torch.from_numpy(trimap_np[b])).numpy().astype(np.float32)

        # round_down_map（-1 表示尚未“掉线”）
        rdm = -np.ones_like(gt_b, dtype=np.float32)

        for i in range(1, len(thresholds)):
            t = thresholds[i]
            t_prev = thresholds[i - 1]

            gt_bin   = (gt_b   >= t)
            pred_bin = (pred_b >= t)
            inter    = (gt_bin & pred_bin).astype(np.uint8)

            if inter.any():
                # LCC of intersection, 4-connectivity (与 skimage connectivity=1 等价)
                num, labels = cv2.connectedComponents(inter, connectivity=4)
                if num > 1:
                    counts = np.bincount(labels.ravel())[1:]  # skip background
                    max_id = counts.argmax() + 1
                    omega  = (labels == max_id).astype(np.uint8)
                else:
                    omega  = np.zeros_like(inter, dtype=np.uint8)
            else:
                omega = np.zeros_like(inter, dtype=np.uint8)

            # 第一次“掉线”的位置，记录为前一个阈值
            drop_mask = (rdm == -1) & (omega == 0)
            rdm[drop_mask] = t_prev

        # 仍为 -1 的像素从未掉线，置为 1
        rdm[rdm == -1] = 1.0

        # φ 映射（仅当差值>=0.15 时起作用）
        gt_diff   = gt_b   - rdm
        pred_diff = pred_b - rdm
        gt_phi   = 1.0 - gt_diff   * (gt_diff   >= 0.15)
        pred_phi = 1.0 - pred_diff * (pred_diff >= 0.15)

        conn_sum = np.sum(np.abs(gt_phi - pred_phi) * roi_mask).astype(np.float32)  # 单图像素和
        errs.append(conn_sum)

    errors = torch.tensor(errs, dtype=torch.float32)  # (B,)
    if scaled:
        errors = errors * 1e-3  # 和 MaGGIe 的 compute_metric 里乘 0.001 对齐

    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        return errors.mean()
    elif reduction == 'sum':
        return errors.sum()
    else:
        raise ValueError("reduction must be 'none' | 'mean' | 'sum'")


def save_visualization(
    rgb, init_mask, gt, trimap,
    ll_cat, hf_cat,
    feats_ll, feats_hf,
    dec_feats_ll, dec_feats_hf,
    recon, recon_raw,
    pred_shallow,
    pred_attn_bn_ll, pred_attn_bn_hf,
    pred_attn_mid_ll, pred_attn_mid_hf,
    wnames,
    save_path,
    up_mode: str = "bilinear",
):
    """
    可视化整条 refiner_xnet 流程（⽣成⾄ save_path）。
    支持任意 wavelet 组合数 L = len(wnames)，全部展开显⽰。
    """
    # ---------------- 基本信息 ---------------- #
    B, _, H, W = rgb.shape
    _, ll_ch, H1, W1 = ll_cat.shape
    L   = len(wnames)                  # wavelet 数
    C   = ll_ch // L                   # 每个 wavelet 的低频通道数
    cols = max(4, L + 1)               # 至少 4 列；>3 时展开到 L+1（+1 给 recon / error）

    # ---------------- 反归⼀化辅助 ---------------- #
    def denorm(x: torch.Tensor) -> torch.Tensor:
        """x∈[-mean/std, (1-mean)/std] → [0,1]"""
        mean = IMAGENET_MEAN.to(x.device)
        std  = IMAGENET_STD.to(x.device)
        return (x * std + mean).clamp(0, 1)

    # ---------------- 准备要画的内容 ---------------- #
    rgb_vis   = denorm(rgb)[0].permute(1, 2, 0).cpu().numpy()
    init_vis  = init_mask[0, 0].cpu().numpy()
    gt_vis    = gt[0, 0].cpu().numpy()
    tri_vis   = trimap[0, 0].cpu().numpy()
    recon_vis = recon[0, 0].detach().cpu().numpy()
    err_vis   = np.abs(gt_vis - recon_vis)

    kernels = torch.tensor([
        [[-0.5,  0.5], [-0.5,  0.5]],  # LH
        [[-0.5, -0.5], [ 0.5,  0.5]],  # HL
        [[ 0.5, -0.5], [-0.5,  0.5]],  # HH
    ], dtype=torch.float32).unsqueeze(1).to(gt.device)
    with torch.no_grad():
        band_tensor = conv_dwt_energy_mask(
            gt.to(torch.float32),        # 保证是 float32
            kernels,
            threshold_quantile=0.75,
            dilation_radius=5,
        )
        # 让 edge 只是骨架（可根据需求改）
        edge_tensor = (band_tensor > 0).float()
        band_np = band_tensor[0,0].cpu().numpy()
        edge_np = edge_tensor[0,0].cpu().numpy()
        overlay = np.stack([band_np, band_np, band_np], axis=-1)
        overlay[edge_np > 0] = [1.0, 0.0, 0.0]

    ll_list = ll_cat.split(C, dim=1)
    hf_list = hf_cat.split(3 * C, dim=1)

    # ---------------- 动态计算⾏数 ---------------- #
    total_enc   = max(len(feats_ll), len(feats_hf))
    enc_rows    = (total_enc + 3) // 4          # 4 列栅格
    total_dec   = max(len(dec_feats_ll), len(dec_feats_hf))
    dec_rows    = (total_dec + 3) // 4
    total_rows  = 4 + enc_rows * 2 + dec_rows * 2   # 头 3 ⾏ + enc 双⾏ + dec 双⾏

    fig, axs = plt.subplots(
        total_rows, cols, figsize=(4 * cols, 3 * total_rows)
    )

    # ========= 第 1 ⾏：RGB / Init / GT / Trimap ========= #
    first_row_imgs = [rgb_vis, init_vis, gt_vis, overlay]
    first_row_titles = ["RGB", "Init Mask", "GT", "DWT Loss Region"]
    for i, (img, ttl) in enumerate(zip(first_row_imgs, first_row_titles)):
        axs[0, i].imshow(img if i == 0 else img, cmap=None if i == 0 else "gray")
        axs[0, i].set_title(ttl)
    for j in range(len(first_row_imgs), cols):
        axs[0, j].axis("off")

    # ========= 第 2 ⾏：各 wavelet LL + Recon ========= #
    for i in range(L):
        ll_up = F.interpolate(
            ll_list[i][:, :1].float(), size=(H, W), mode=up_mode
        )[0, 0].cpu()
        axs[1, i].imshow(ll_up, cmap="gray")
        axs[1, i].set_title(f"{wnames[i]} LL")
    axs[1, cols - 2].imshow(recon_raw[0,0], cmap="gray")
    axs[1, cols - 2].set_title("Recon (Raw)")
    axs[1, cols - 1].imshow(recon_vis, cmap="gray")
    axs[1, cols - 1].set_title("Recon (IWT)")
    for j in range(L, cols - 1):
        axs[1, j].axis("off")          # 介于 L 和 Recon 之间的空位

    # ========= 第 3 ⾏：各 wavelet HF + ErrMap ========= #
    for i in range(L):
        # 展开出 (B, C, 3, H/2, W/2)
        hf_i = hf_list[i].view(B, C, 3, H1, W1).float()

        # 计算能量图，而非简单平均
        # +1e-12 是为了数值稳定，避免 sqrt(0)
        hf_energy = torch.sqrt((hf_i ** 2).sum(dim=2) + 1e-12)   # -> (B,C,H/2,W/2)

        # 仅取可视化用的第 1 个通道，再上采样回原分辨率
        hf_up = F.interpolate(hf_energy[:, :1], size=(H, W), mode=up_mode)[0, 0]

        # 自适应 vmax，避免被极大边缘 “压暗”
        vmax = hf_up.abs().quantile(0.995).item() + 1e-8
        axs[2, i].imshow(hf_up.cpu(), cmap="bwr", vmin=-vmax, vmax=vmax)
        axs[2, i].set_title(f"{wnames[i]} HF")
    axs[2, cols - 1].imshow(err_vis, cmap="hot")
    axs[2, cols - 1].set_title("GT – Recon Err")
    for j in range(L, cols - 1):
        axs[2, j].axis("off")

    # ========= 第 4 行：Auxiliary Heads 输出 ========= #
    aux_row = 3
    # 五个辅助输出按列依次展示
    titles = ["Att BN LL", "Attn BN HF", "Attn Mid LL", "Attn Mid HF"]
    images = [pred_attn_bn_ll, pred_attn_bn_hf, pred_attn_mid_ll, pred_attn_mid_hf]
    for i in range(len(titles)):
        # 上采样到原始大小，取第 0 张图第 0 通道
        m = F.interpolate(images[i].float(), size=(H, W), mode=up_mode)[0, 0].cpu().numpy()
        axs[aux_row, i].imshow(m, cmap="gray")
        axs[aux_row, i].set_title(titles[i])
    # 隐藏多余列
    for j in range(len(images), cols):
        axs[aux_row, j].axis("off")

    # ========= Encoder 可视化（两⾏为⼀组） ========= #
    base_row = 4
    for idx in range(total_enc):
        r_ll = base_row + (idx // 4) * 2
        r_hf = r_ll + 1
        c    = idx % 4
        # LL
        if idx < len(feats_ll):
            fmap = feats_ll[idx].mean(1)[0]
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_ll, c].imshow(norm.cpu(), cmap="viridis")
            axs[r_ll, c].set_title(f"Enc LL L{idx}")
        else:
            axs[r_ll, c].axis("off")
        # HF
        if idx < len(feats_hf):
            fmap = feats_hf[idx].mean(1)[0]
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_hf, c].imshow(norm.cpu(), cmap="viridis")
            axs[r_hf, c].set_title(f"Enc HF L{idx}")
        else:
            axs[r_hf, c].axis("off")
        # 其余列隐藏
        for j in range(4, cols):
            axs[r_ll, j].axis("off")
            axs[r_hf, j].axis("off")

    # ========= Decoder 可视化（两⾏为⼀组） ========= #
    dec_base = base_row + enc_rows * 2
    for idx in range(total_dec):
        r_ll = dec_base + (idx // 4) * 2
        r_hf = r_ll + 1
        c    = idx % 4
        # LL
        if idx < len(dec_feats_ll):
            fmap = dec_feats_ll[idx][0].mean(0)
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_ll, c].imshow(norm.cpu(), cmap="magma")
            axs[r_ll, c].set_title(f"Dec LL L{idx}")
        else:
            axs[r_ll, c].axis("off")
        # HF
        if idx < len(dec_feats_hf):
            fmap = dec_feats_hf[idx][0].mean(0)
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_hf, c].imshow(norm.cpu(), cmap="magma")
            axs[r_hf, c].set_title(f"Dec HF L{idx}")
        else:
            axs[r_hf, c].axis("off")
        for j in range(4, cols):
            axs[r_ll, j].axis("off")
            axs[r_hf, j].axis("off")

    # ========= 收尾 ========= #
    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def train_one_epoch(epoch, loader):
    model.train()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    log_path = os.path.join(cfg['log_dir'], 'train_metrics.jsonl')

    for step, batch in enumerate(loader):
        rgb, init_mask, gt, trimap = batch  # stay on CPU; DataParallel scatter
        # print(f"At step: {step}... \n")
        need_viz = (step % cfg['viz_interval'] == 0)
        is_update_step = ((step + 1)%cfg["accum_step"]==0) or (step+1==len(loader))
        edge_k_cur = get_edge_k(epoch)
        # rgb, init_mask, gt, trimap = [x.to(device) for x in batch]
        # rgb, init_mask, gt, trimap = [x.pin_memory().to('cuda', non_blocking=True) for x in batch]

        # with autocast(enabled=(device_type == "cuda")):
        feats_ll, feats_hf, ll_cat, hf_cat, dec_feats_ll, \
        dec_feats_hf, shallow_ds, attn_bn_ll, attn_bn_hf, \
        attn_mid_ll, attn_mid_hf, recon_raw, recon = model(rgb, init_mask)

        
        if need_viz:
            feats_ll_cpu  = [f.detach().cpu() for f in feats_ll]
            feats_hf_cpu  = [f.detach().cpu() for f in feats_hf]
            dec_ll_cpu    = [f.detach().cpu() for f in dec_feats_ll]
            dec_hf_cpu    = [f.detach().cpu() for f in dec_feats_hf]
            ll_cat_cpu    = ll_cat.detach().cpu() 
            hf_cat_cpu    = hf_cat.detach().cpu() 
            recon_raw_cpu = recon_raw.detach().cpu()
        del feats_ll, feats_hf, dec_feats_ll, dec_feats_hf, recon_raw
        torch.cuda.empty_cache()

        # 主干损失
        if gt.device != recon.device:         
            gt      = gt.to(recon.device, non_blocking=True)
            trimap  = trimap.to(recon.device, non_blocking=True)
        gt = gt.to(recon.dtype) 
        l1_full = loss_l1(recon, gt)
        hf_full, hf_band = dwt_loss_fn(recon, gt)

        # ========== Attention BN 监督 ==========
        alpha_attn_down = F.interpolate(gt, size=attn_bn_ll.shape[-2:], mode='bilinear')
        attn_l1 = loss_l1(attn_bn_ll, alpha_attn_down) + loss_l1(attn_bn_hf, alpha_attn_down)
        hf_ll_full, hf_ll_band = dwt_loss_fn(attn_bn_ll, alpha_attn_down)
        hf_hf_full, hf_hf_band = dwt_loss_fn(attn_bn_hf, alpha_attn_down)

        # ========== Attention Mid 监督 ==========
        alpha_mid_attn_down = F.interpolate(gt, size=attn_mid_ll.shape[-2:], mode='bilinear')
        attn_mid_l1 = loss_l1(attn_mid_ll, alpha_mid_attn_down) + loss_l1(attn_mid_hf, alpha_mid_attn_down)
        hf_ll_mid_full, hf_ll_mid_band = dwt_loss_fn(attn_mid_ll, alpha_mid_attn_down)
        hf_hf_mid_full, hf_hf_mid_band = dwt_loss_fn(attn_mid_hf, alpha_mid_attn_down)

        # 合并两路 HF loss
        hf_attn_full = (hf_ll_full + hf_hf_full + hf_ll_mid_full + hf_hf_mid_full)
        hf_attn_band = None
        if hf_ll_band is not None:
            hf_attn_band = (hf_ll_band + hf_hf_band + hf_ll_mid_band + hf_hf_mid_band)
        
        attn_loss = 0.05*attn_l1 + 0.05*attn_mid_l1 + 0.25*hf_attn_full
        if hf_attn_band is not None:
            attn_loss += 0.25 * hf_attn_band

        """# ========== Shallow 监督 ==========
        alpha_shallow_down = F.interpolate(gt, size=shallow_ds.shape[-2:], mode='bilinear')
        sh_l1   = loss_l1(shallow_ds, alpha_shallow_down)
        hf_sh_full, hf_sh_band = dwt_loss_fn(shallow_ds, alpha_shallow_down)
        shallow_aux_loss = 0.2*sh_l1 + 0.4*hf_sh_full
        if hf_sh_band is not None:
            shallow_aux_loss += 0.4 * hf_sh_band"""

        loss = (
            0.5 * l1_full +
            2.5 * hf_full +
            (2.5 * hf_band if hf_band is not None else 0.0) +
            attn_loss 
            # shallow_aux_loss
        ) / cfg["accum_step"]

        with torch.no_grad():
            p_dbg = next((p for p in model.parameters() if p.grad is not None), None)
            if p_dbg is not None:
                grad_mu_before  = p_dbg.grad.float().mean().item()
                grad_min_before = p_dbg.grad.float().min().item()
                grad_max_before = p_dbg.grad.float().max().item()
                print(f"[debug BEFORE] grad μ={grad_mu_before:.6e}, "
                    f"min={grad_min_before:.6e}, max={grad_max_before:.6e}")
            else:
                print("[debug BEFORE] No gradients found before backward.")

        # 1) 如果 loss 本身不是有限值，直接跳过
        if not torch.isfinite(loss):
            print(f"[e{epoch} s{step}] ⚠️ loss={loss.item()} 非有限，跳过")
            optimizer.zero_grad(set_to_none=True)
            # scaler.update()
            continue

        try:
            # scaler.scale(loss).backward()
            loss.backward()
            # ----- Debug Grad Info -----
            with torch.no_grad():
                grads = [p for p in model.parameters() if p.grad is not None]
                if len(grads) == 0:
                    print("[debug AFTER] No parameters have gradients!")
                else:
                    p_dbg = grads[0]
                    grad_mu_after  = p_dbg.grad.float().mean().item()
                    grad_min_after = p_dbg.grad.float().min().item()
                    grad_max_after = p_dbg.grad.float().max().item()
                    print(f"[debug AFTER] grad μ={grad_mu_after:.6e}, "
                        f"min={grad_min_after:.6e}, max={grad_max_after:.6e}")
        except RuntimeError as e:
            msg = str(e).lower()
            if "nan" in msg or "inf" in msg:
                optimizer.zero_grad(set_to_none=True)
                print(f"[e{epoch} s{step}] ⚠️ backward 遇到 NaN/Inf，跳过 micro‑batch")
                # scaler.update()
                continue
            else:
                raise

        # ============================================================

        # === 可视化并保存图像 ===
        if need_viz:
            save_visualization(
                rgb.detach().cpu(), init_mask.detach().cpu(), 
                gt.detach().to(recon.device).to(recon.dtype), trimap.detach().cpu(),
                ll_cat_cpu, hf_cat_cpu,
                feats_ll_cpu,
                feats_hf_cpu,
                dec_ll_cpu,
                dec_hf_cpu,
                recon.detach().cpu(),
                recon_raw_cpu,
                shallow_ds.detach().cpu(),
                attn_bn_ll.detach().cpu(), attn_bn_hf.detach().cpu(),
                attn_mid_ll.detach().cpu(), attn_mid_hf.detach().cpu(),
                cfg['wavelet_list'],
                os.path.join(cfg['vis_dir'], f"e{epoch}_{int(step)}.png")
            )

        if is_update_step:
            print(
                f"[e{epoch} s{step}] "
                f"L1={l1_full.item()*0.5:.4f} | "
                f"hf_full={hf_full.item()*2.5:.4f} | "
                f"hf_band={hf_band.item()*2.5:.4f} | "
                f"attn={attn_loss.item():.4f} | "
                # f"shallow_aux={shallow_aux_loss.item():.4f} | "
                f"Total={(loss.item()*cfg['accum_step']):.4f}"
            )

            # === 梯度正常，执行优化器 step 和日志记录 ===
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            """scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)"""
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # === 记录日志 ===
            record = {
                "epoch":              int(epoch),
                "step":               int(step),
                "L1_raw":             float(l1_full.item()),
                "L1_weighted":        float(l1_full.item()*0.5),
                "Hf_full_raw":        float(hf_full.item()),
                "Hf_band_raw":        float(hf_band.item()),
                "Hf_full_weighted":   float(hf_full.item()*2.5),
                "Hf_band_weighted":   float(hf_band.item()*2.5),
                "attn_loss":          float(attn_loss.item()),
                "attn_bn_L1":         float(attn_l1.item()*0.05),
                "attn_bn_Hf_full":    float(hf_attn_full.item()*0.25),
                "attn_bn_Hf_band":    float(hf_attn_band.item()*0.25),
                "attn_mid_L1":        float(attn_mid_l1.item()*0.05),
                # "shallow_loss":       float(shallow_aux_loss.item()),
                # "shallow_L1":         float(sh_l1 .item()*0.2),
                # "shallow_Hf_full":    float(hf_sh_full.item()*0.4),
                # "shallow_Hf_band":    float(hf_sh_band.item()*0.4),
                "total_loss":         float(loss.item() * cfg["accum_step"]),
            }

            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        del recon
        del rgb, init_mask, gt, trimap                 # ← 把输入也扔掉
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()                       # 可选：再清 IPC 缓存


@torch.no_grad()
def validate(model, val_loader, epoch, meters):
    model.eval()
    for meter in meters.values():
        meter.reset()
    log_path = os.path.join(cfg['log_dir'], 'val_metrics.jsonl')

    val_loss_sum, val_count = 0.0, 0

    for step, batch in enumerate(val_loader):
        rgb, init_mask, gt, trimap = batch
        device = next(model.parameters()).device
        rgb, init_mask, gt, trimap = rgb.to(device), init_mask.to(device), gt.to(device), trimap.to(device)

        need_viz = (step % int(cfg['viz_interval']) == 0)

        feats_ll, feats_hf, ll_cat, hf_cat, dec_feats_ll, \
        dec_feats_hf, shallow_ds, attn_bn_ll, attn_bn_hf, \
        attn_mid_ll, attn_mid_hf, recon_raw, recon = model(rgb, init_mask)

        if need_viz:
            feats_ll_cpu  = [f.detach().cpu() for f in feats_ll]
            feats_hf_cpu  = [f.detach().cpu() for f in feats_hf]
            dec_ll_cpu    = [f.detach().cpu() for f in dec_feats_ll]
            dec_hf_cpu    = [f.detach().cpu() for f in dec_feats_hf]
            ll_cat_cpu    = ll_cat.detach().cpu() 
            hf_cat_cpu    = hf_cat.detach().cpu() 
            recon_raw_cpu = recon_raw.detach().cpu()
        del feats_ll, feats_hf, dec_feats_ll, dec_feats_hf, recon_raw
        torch.cuda.empty_cache()

        # ====== 计算与训练一致的验证损失 ======
        gt_f = gt.to(recon.device, non_blocking=True).to(recon.dtype)

        # 主干
        l1_full = loss_l1(recon, gt_f)
        hf_full, hf_band = dwt_loss_fn(recon, gt_f)

        """# Attention BN
        alpha_attn_down = F.interpolate(gt_f, size=attn_bn_ll.shape[-2:], mode='bilinear', align_corners=False)
        attn_l1 = loss_l1(attn_bn_ll, alpha_attn_down) + loss_l1(attn_bn_hf, alpha_attn_down)
        hf_ll_full, hf_ll_band = dwt_loss_fn(attn_bn_ll, alpha_attn_down)
        hf_hf_full, hf_hf_band = dwt_loss_fn(attn_bn_hf, alpha_attn_down)

        # Attention Mid
        alpha_mid_attn_down = F.interpolate(gt_f, size=attn_mid_ll.shape[-2:], mode='bilinear', align_corners=False)
        attn_mid_l1 = loss_l1(attn_mid_ll, alpha_mid_attn_down) + loss_l1(attn_mid_hf, alpha_mid_attn_down)
        hf_ll_mid_full, hf_ll_mid_band = dwt_loss_fn(attn_mid_ll, alpha_mid_attn_down)
        hf_hf_mid_full, hf_hf_mid_band = dwt_loss_fn(attn_mid_hf, alpha_mid_attn_down)

        hf_attn_full = (hf_ll_full + hf_hf_full + hf_ll_mid_full + hf_hf_mid_full)
        hf_attn_band = (hf_ll_band + hf_hf_band + hf_ll_mid_band + hf_hf_mid_band) if hf_ll_band is not None else None

        attn_loss = 0.05*attn_l1 + 0.05*attn_mid_l1 + 0.25*hf_attn_full
        if hf_attn_band is not None:
            attn_loss += 0.25*hf_attn_band

        val_loss_batch = (
            0.5 * l1_full +
            2.5 * hf_full +
            (2.5 * hf_band if hf_band is not None else 0.0) +
            attn_loss
        )

        # 累计（用样本数加权平均更稳）
        bsz = rgb.size(0)
        val_loss_sum += float(val_loss_batch.item()) * bsz
        val_count    += bsz"""

        # ====== 指标（对齐 MaGGIe / MatAnyOne） ======
        alpha_pred = recon.clamp(0.0, 1.0)

        # 1) MAD (full): per-image mean(|err|)*1000  -> (B,)
        mad_full_vec = compute_mad_full(alpha_pred, gt)  # 返回 (B,)

        # 2) SAD (unknown): per-image sum(|err|), 显示缩放 /1000 -> (B,)
        sad_unknown_vec = compute_sad(alpha_pred, gt, trimap,
                                      reduction='none',  # 逐图向量
                                      scaled=True)       # /1000，若不想缩放改 False

        # 3) MSE (unknown, MaGGIe): 返回 per-image 向量，内部 *1e10
        mse_vec = compute_mse_unknown(alpha_pred, gt, trimap, reduction='none')

        # 4) Grad (unknown, MaGGIe): 高斯导数梯度，平方差像素和（不缩放），per-image
        grad_vec = compute_gradient_error_maggie(alpha_pred, gt, trimap, reduction='none')

        # 5) Conn (unknown, MaGGIe): per-image 像素和 ×1e-3
        conn_vec = compute_connectivity_error_maggie(alpha_pred, gt, trimap,
                                                     step=0.1, reduction='none', scaled=True)

        # ====== 累计到 meters（逐图等权平均：sum(vec), n=len(vec)） ======
        meters['mad_full'].update(mad_full_vec.sum().item(), n=mad_full_vec.numel())
        meters['sad_unknown'].update(sad_unknown_vec.sum().item(), n=sad_unknown_vec.numel())
        meters['mse'].update(mse_vec.sum().item(), n=mse_vec.numel())
        meters['grad'].update(grad_vec.sum().item(), n=grad_vec.numel())
        meters['conn'].update(conn_vec.sum().item(), n=conn_vec.numel())

        if need_viz:
            save_visualization(
                rgb.detach().cpu(), init_mask.detach().cpu(), 
                gt.detach().to(recon.device).to(recon.dtype), trimap.detach().cpu(),
                ll_cat_cpu, hf_cat_cpu,
                feats_ll_cpu,
                feats_hf_cpu,
                dec_ll_cpu,
                dec_hf_cpu,
                recon.detach().cpu(),
                recon_raw_cpu,
                shallow_ds.detach().cpu(),
                #attn_bn_ll.detach().cpu(), attn_bn_hf.detach().cpu(),
                #attn_mid_ll.detach().cpu(), attn_mid_hf.detach().cpu(),
                torch.zeros(rgb.shape).detach().cpu(), torch.zeros(rgb.shape).detach().cpu(),
                torch.zeros(rgb.shape).detach().cpu(), torch.zeros(rgb.shape).detach().cpu(),
                cfg['wavelet_list'],
                os.path.join(cfg['vis_val_dir'], f"e{epoch}_{step}.png")
            )

        # 清理
        del recon, rgb, init_mask, gt, trimap
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 打印与记录（保持不变）
    print(
        f"[e{epoch} Validation] "
        f"mad full={meters['mad_full'].avg:.4f} | "
        f"sad unknown={meters['sad_unknown'].avg:.4f} | "
        f"mse={meters['mse'].avg:.4f} | "
        f"grad={meters['grad'].avg:.4f} | "
        f"conn={meters['conn'].avg:.4f} | "
    )
    record = {
        "epoch": int(epoch),
        "mad full": float(meters['mad_full'].avg),
        "sad unknown": float(meters['sad_unknown'].avg),
        "mse": float(meters['mse'].avg),
        "grad": float(meters['grad'].avg),
        "conn": float(meters['conn'].avg),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 返回与训练一致的验证损失
    val_loss_avg = val_loss_sum / max(1, val_count)
    return val_loss_avg


if __name__ == '__main__':
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    os.makedirs(cfg['vis_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    main_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = list(range(torch.cuda.device_count()))   # 在 4,5 可见下 => [0,1]

    # --- 构建 DWT 损失 ---
    dwt1 = DWTForward(J=1, wave=cfg['wavelet_list'][0], mode='zero').to(device)
    dwt_loss_fn = DWTLoss(
        hf_weighted=True,          # Sobel 加权
        enable_band=True,          # 启用 HF-energy 带状
        band_radius=5,             # 膨胀半径 ±2px
        band_weight=3,           # 带内再 ×2.5
        eps = 1e-6,
        threshold_quantile=0.75,   # 只取前 10% 能量
    )

    # --- 构建模型 ---
    model = refiner_xnet(
        wavelet_list=cfg['wavelet_list'],
        base_channels=cfg['base_channel'],
        num_layers=4,
        blocks_per_layer=3,
        d_head=cfg['d_head'],
        modes_bn={'row','col','diag','global'},
        modes_mid={'row','col','diag','global'},
    )
    print("before:", count_bn(model), "BN,", count_gn(model), "GN")
    bn_to_gn(model)
    print("after :", count_bn(model), "BN,", count_gn(model), "GN")

    model = model.to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    # --- 优化器：前 3 个 epoch shallow_lr = 0.1×，以后恢复 1× ---
    core = model.module  # DP 下实际模型
    post_params = list(core.post.parameters())
    for p in post_params:
        p.requires_grad_(False)   # E0 冻结
    base_params = [p for n, p in core.named_parameters() if not n.startswith("post.")]
    optimizer = optim.AdamW([
        {"params": base_params, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "name": "base"},
        {"params": post_params, "lr": 0.0, "weight_decay": cfg["weight_decay"], "name": "post"},
    ])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=2, threshold=1e-4,
        min_lr=1e-7, verbose=True
    )
    scaler = GradScaler(enabled=(device_type=='cuda'))

    resume_epoch = 19   # <-- 在此修改为你想从哪个 epoch 恢复
    resume_path  = os.path.join(cfg['checkpoint_dir'], f'epoch{resume_epoch}.pth')

    if os.path.isfile(resume_path):
        print(f"=> Loading checkpoint for epoch {resume_epoch} from {resume_path}")
        state_dict = torch.load(resume_path, map_location='cuda:0')
        model.load_state_dict(state_dict)
        start_epoch = resume_epoch + 1
        print(f"=> Resuming from epoch {resume_epoch}, next start at {start_epoch}")
    else:
        print("=> No valid resume checkpoint found, training from scratch")
        start_epoch = 0

    # --- 数据加载 ---
    low_loader, low_val_loader = build_dataloaders(
        cfg['csv_path'],
        (720, 1280),
        cfg['batch_size'],
        num_workers=16,
        split_ratio=0.8,
        seed=cfg['seed'],
        sample_fraction=0.5,
        do_crop=False,
    )

    # --- 指标 ---
    meters = {
        'mad_full': AvgMeter(),
        'sad_unknown': AvgMeter(),
        'mse': AvgMeter(),
        'grad': AvgMeter(),
        'conn': AvgMeter(),
    }

    # --- 训练主循环 ---
    SHALLOW_WARMUP_FREEZE_EPOCHS = 2   # E0 冻结
    SHALLOW_WARMUP_LOWLR_EPOCHS  = 4   # E1–E2 低LR，E3起提到1×

    for epoch in range(start_epoch, cfg['num_epochs']):
        print(f"[epoch {epoch}] enter loop", flush=True)  # ← 看看是否真的进来了

        if epoch == SHALLOW_WARMUP_FREEZE_EPOCHS:
            for p in post_params:
                p.requires_grad_(True)
            for g in optimizer.param_groups:
                if g.get("name") == "post":
                    g["lr"] = cfg["lr"] * 0.1
            print(f"[lr schedule] epoch {epoch}: unfreeze post, lr -> {cfg['lr']*0.1:.1e}")

        # E1/2→E3：提升到 1×
        if epoch == SHALLOW_WARMUP_LOWLR_EPOCHS:
            for g in optimizer.param_groups:
                if g.get("name") == "post":
                    g["lr"] = cfg["lr"]
            print(f"[lr schedule] epoch {epoch}: post lr -> {cfg['lr']:.1e}")

        # train_one_epoch(epoch, low_loader)
        val_loss = validate(model, low_val_loader, epoch, meters)
        if epoch >= SHALLOW_WARMUP_LOWLR_EPOCHS:
            scheduler.step(val_loss)
        else:
            print(f"[plateau] skip at epoch {epoch} (warmup)")
        ckpt = os.path.join(cfg['checkpoint_dir'], f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
        gc.collect()
        torch.cuda.empty_cache()

    print('✅ Training complete!')


