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

# enable anomaly detection to locate the first NaN/Inf during backprop
torch.autograd.set_detect_anomaly(True)

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_xnet_rgbembed import refiner_xnet
from pipeline.metrics import *
from pipeline.utils import *

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ================= warmup and other config =================
WARM_EPOCHS        = 2         # Sobel 权重 warm‑up 轮数
TARGET_FRAC        = 0.15      # Sobel_raw 占主干损失期望比例
MAX_SOBEL_GAIN     = 2.0       # Sobel 整体最大放大倍数
GAIN_RAMP_EPOCHS   = 10        # gain 从 1→MAX 的线性 epoch 数

# model config
cfg = {
    'num_epochs':       21,
    'batch_size':       2,
    "accum_step":       4,
    'checkpoint_dir':   'checkpoints_xnet_db1_final',
    'log_dir':          'log_xnet_db1_final',
    'csv_path':         '../data/pair_for_refiner.csv',
    'vis_dir':          'vis_xnet_db1_final_train',
    'vis_val_dir':      'vis_xnet_db1_final_train_test',
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


def loss_l1(alpha_pred: torch.Tensor,
            alpha_gt:   torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        diff = (alpha_pred - alpha_gt).abs() * mask
        loss = diff.sum() / (mask.sum() + 1e-8)   # 避免除 0
    else:
        loss = F.l1_loss(alpha_pred, alpha_gt)
    return loss


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
        # Haar high pass filter
        kernels = torch.tensor([
            [[-0.5,  0.5], [-0.5,  0.5]],  # LH
            [[-0.5, -0.5], [ 0.5,  0.5]],  # HL
            [[ 0.5, -0.5], [-0.5,  0.5]],  # HH
        ], dtype=torch.float32)  # shape (3, 2, 2)
        self.register_buffer('kernels', kernels.unsqueeze(1))  # → (3,1,2,2)

        self.hf_weighted = hf_weighted
        self.enable_band = enable_band
        self.band_radius = band_radius
        self.band_weight = band_weight
        self.eps = eps
        self.th_q = threshold_quantile

    def forward(
        self,
        alpha_pred: torch.Tensor,  # (B,1,H,W)
        alpha_gt:   torch.Tensor,  # (B,1,H,W)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # HF diff
        kernels = self.kernels.to(dtype=alpha_pred.dtype, device=alpha_pred.device)
        hp_pred = F.conv2d(alpha_pred, kernels, padding=1)
        hp_gt = F.conv2d(alpha_gt,   kernels, padding=1)
        diff = (hp_pred - hp_gt).abs()
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=1e6)

        # Sobel edge weighting
        if self.hf_weighted:
            w = sobel_edge_weight(alpha_gt)
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


def train_one_epoch(epoch, loader):
    model.train()
    log_path = os.path.join(cfg['log_dir'], 'train_metrics.jsonl')

    for step, batch in enumerate(loader):
        rgb, init_mask, gt, trimap = batch
        need_viz = (step % cfg['viz_interval'] == 0)
        is_update_step = ((step + 1)%cfg["accum_step"]==0) or (step+1==len(loader))

        # with autocast(enabled=(device_type == "cuda")):
        # disable amp due to some env issue on current machine
        feats_ll, feats_hf, ll_cat, hf_cat, dec_feats_ll, \
        dec_feats_hf, shallow_ds, attn_bn_ll, attn_bn_hf, \
        attn_mid_ll, attn_mid_hf, recon_raw, recon = model(rgb, init_mask)

        if need_viz:
            feats_ll_cpu = [f.detach().cpu() for f in feats_ll]
            feats_hf_cpu = [f.detach().cpu() for f in feats_hf]
            dec_ll_cpu = [f.detach().cpu() for f in dec_feats_ll]
            dec_hf_cpu = [f.detach().cpu() for f in dec_feats_hf]
            ll_cat_cpu = ll_cat.detach().cpu()
            hf_cat_cpu = hf_cat.detach().cpu()
            recon_raw_cpu = recon_raw.detach().cpu()
        del feats_ll, feats_hf, dec_feats_ll, dec_feats_hf, recon_raw
        torch.cuda.empty_cache()

        # loss on main model path
        if gt.device != recon.device:         
            gt = gt.to(recon.device, non_blocking=True)
            trimap = trimap.to(recon.device, non_blocking=True)
        gt = gt.to(recon.dtype) 
        l1_full = loss_l1(recon, gt)
        hf_full, hf_band = dwt_loss_fn(recon, gt)

        # ========== Attention BN auxilary head ==========
        alpha_attn_down = F.interpolate(gt, size=attn_bn_ll.shape[-2:], mode='bilinear')
        attn_l1 = loss_l1(attn_bn_ll, alpha_attn_down) + loss_l1(attn_bn_hf, alpha_attn_down)
        hf_ll_full, hf_ll_band = dwt_loss_fn(attn_bn_ll, alpha_attn_down)
        hf_hf_full, hf_hf_band = dwt_loss_fn(attn_bn_hf, alpha_attn_down)

        # ========== Attention Mid auxilary head ==========
        alpha_mid_attn_down = F.interpolate(gt, size=attn_mid_ll.shape[-2:], mode='bilinear')
        attn_mid_l1 = loss_l1(attn_mid_ll, alpha_mid_attn_down) + loss_l1(attn_mid_hf, alpha_mid_attn_down)
        hf_ll_mid_full, hf_ll_mid_band = dwt_loss_fn(attn_mid_ll, alpha_mid_attn_down)
        hf_hf_mid_full, hf_hf_mid_band = dwt_loss_fn(attn_mid_hf, alpha_mid_attn_down)

        # add loss
        hf_attn_full = (hf_ll_full + hf_hf_full + hf_ll_mid_full + hf_hf_mid_full)
        hf_attn_band = None
        if hf_ll_band is not None:
            hf_attn_band = (hf_ll_band + hf_hf_band + hf_ll_mid_band + hf_hf_mid_band)
        
        attn_loss = 0.05*attn_l1 + 0.05*attn_mid_l1 + 0.25*hf_attn_full
        if hf_attn_band is not None:
            attn_loss += 0.25 * hf_attn_band

        loss = (
            0.5 * l1_full +
            2.5 * hf_full +
            (2.5 * hf_band if hf_band is not None else 0.0) +
            attn_loss
        ) / cfg["accum_step"]  # accumulation

        with torch.no_grad():
            p_dbg = next((p for p in model.parameters() if p.grad is not None), None)
            if p_dbg is not None:
                grad_mu_before = p_dbg.grad.float().mean().item()
                grad_min_before = p_dbg.grad.float().min().item()
                grad_max_before = p_dbg.grad.float().max().item()
                print(f"[debug BEFORE] grad μ={grad_mu_before:.6e}, "
                    f"min={grad_min_before:.6e}, max={grad_max_before:.6e}")
            else:
                print("[debug BEFORE] No gradients found before backward.")

        if not torch.isfinite(loss):
            print(f"[e{epoch} s{step}] ⚠️ loss={loss.item()} 非有限，跳过")
            optimizer.zero_grad(set_to_none=True)
            continue

        try:
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
        if need_viz:
            save_visualization(
                rgb.detach().cpu(), init_mask.detach().cpu(), 
                gt.detach().to(recon.device).to(recon.dtype), trimap.detach().cpu(),
                ll_cat_cpu, hf_cat_cpu, feats_ll_cpu, feats_hf_cpu,
                dec_ll_cpu, dec_hf_cpu,
                recon.detach().cpu(), recon_raw_cpu,
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
                f"Total={(loss.item()*cfg['accum_step']):.4f}"
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # === record ===
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
                "total_loss":         float(loss.item() * cfg["accum_step"]),
            }

            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        del recon
        del rgb, init_mask, gt, trimap
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
            feats_ll_cpu = [f.detach().cpu() for f in feats_ll]
            feats_hf_cpu = [f.detach().cpu() for f in feats_hf]
            dec_ll_cpu = [f.detach().cpu() for f in dec_feats_ll]
            dec_hf_cpu = [f.detach().cpu() for f in dec_feats_hf]
            ll_cat_cpu = ll_cat.detach().cpu()
            hf_cat_cpu = hf_cat.detach().cpu()
            recon_raw_cpu = recon_raw.detach().cpu()
        del feats_ll, feats_hf, dec_feats_ll, dec_feats_hf, recon_raw
        torch.cuda.empty_cache()

        gt_f = gt.to(recon.device, non_blocking=True).to(recon.dtype)

        # loss on main model path
        l1_full = loss_l1(recon, gt_f)
        hf_full, hf_band = dwt_loss_fn(recon, gt_f)

        # Attention BN
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

        # accumulate (sample-weighted average is more stable)
        bsz = rgb.size(0)
        val_loss_sum += float(val_loss_batch.item()) * bsz
        val_count += bsz

        alpha_pred = recon.clamp(0.0, 1.0)

        # 1) MAD (full): per-image mean(|err|)*1000  -> (B,)
        mad_full_vec = compute_mad_full(alpha_pred, gt)  # (B,)

        # 2) SAD (unknown): per-image sum(|err|)/1000 -> (B,)
        sad_unknown_vec = compute_sad(alpha_pred, gt, trimap,
                                      reduction='none',
                                      scaled=True)

        # 3) MSE (unknown, MaGGIe): per-image vec*1e10
        mse_vec = compute_mse_unknown(alpha_pred, gt, trimap, reduction='none')

        # 4) Grad (unknown, MaGGIe):
        # Gaussian derivative gradient, pixel-wise squared difference sum (no scaling)
        grad_vec = compute_gradient_error_maggie(alpha_pred, gt, trimap, reduction='none')

        # 5) Conn (unknown, MaGGIe): per-image sum * 1e-3
        conn_vec = compute_connectivity_error_maggie(alpha_pred, gt, trimap,
                                                     step=0.1, reduction='none', scaled=True)

        # ====== update meters with per-image equal weighting: sum(vec), n=len(vec) ======
        meters['mad_full'].update(mad_full_vec.sum().item(), n=mad_full_vec.numel())
        meters['sad_unknown'].update(sad_unknown_vec.sum().item(), n=sad_unknown_vec.numel())
        meters['mse'].update(mse_vec.sum().item(), n=mse_vec.numel())
        meters['grad'].update(grad_vec.sum().item(), n=grad_vec.numel())
        meters['conn'].update(conn_vec.sum().item(), n=conn_vec.numel())

        if need_viz:
            save_visualization(
                rgb.detach().cpu(), init_mask.detach().cpu(), 
                gt.detach().to(recon.device).to(recon.dtype), trimap.detach().cpu(),
                ll_cat_cpu, hf_cat_cpu, feats_ll_cpu, feats_hf_cpu,
                dec_ll_cpu, dec_hf_cpu,
                recon.detach().cpu(), recon_raw_cpu,
                shallow_ds.detach().cpu(),
                attn_bn_ll.detach().cpu(), attn_bn_hf.detach().cpu(),
                attn_mid_ll.detach().cpu(), attn_mid_hf.detach().cpu(),
                cfg['wavelet_list'],
                os.path.join(cfg['vis_val_dir'], f"e{epoch}_{step}.png")
            )

        del recon, rgb, init_mask, gt, trimap
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # print and log record
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

    # --- DWT loss func ---
    dwt1 = DWTForward(J=1, wave=cfg['wavelet_list'][0], mode='zero').to(device)
    dwt_loss_fn = DWTLoss(
        hf_weighted=True,
        enable_band=True,          # enable HF-energy band
        band_radius=5,             # dilation radius ±5px
        band_weight=3,           # band width ×3.0
        eps=1e-6,
        threshold_quantile=0.75,   # take highest 25% energy only
    )

    # --- construct model ---
    model = refiner_xnet(
        wavelet_list=cfg['wavelet_list'],
        base_channels=cfg['base_channel'],
        num_layers=4,
        blocks_per_layer=3,
        d_head=cfg['d_head'],
        modes_bn={'row','col','diag','global'},
        modes_mid={'row','col','diag','global'},
    )
    bn_to_gn(model)

    # to device and parallel
    model = model.to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    # --- optimizer schedule: shallow_lr 0.1× for warmup epochs, then back to 1× ---
    core = model.module
    post_params = list(core.post.parameters())
    for p in post_params:
        p.requires_grad_(False)   # freeze post to stablize other components first
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

    resume_epoch = 19   # where to resume training/validation
    resume_path = os.path.join(cfg['checkpoint_dir'], f'epoch{resume_epoch}.pth')

    if os.path.isfile(resume_path):
        print(f"=> Loading checkpoint for epoch {resume_epoch} from {resume_path}")
        state_dict = torch.load(resume_path, map_location='cuda:0')
        model.load_state_dict(state_dict)
        start_epoch = resume_epoch + 1
        print(f"=> Resuming from epoch {resume_epoch}, next start at {start_epoch}")
    else:
        print("=> No valid resume checkpoint found, training from scratch")
        start_epoch = 0

    # --- data loader ---
    low_loader, low_val_loader = build_dataloaders(
        cfg['csv_path'], (720, 1280), cfg['batch_size'],
        num_workers=16, split_ratio=0.8, seed=cfg['seed'],
        sample_fraction=0.5, do_crop=False,
    )
    """
    # for batch testing:
    low_loader, low_val_loader = build_dataloaders(
        cfg['csv_path'], (720, 1280), cfg['batch_size'],
        num_workers=16, split_ratio=0.0, seed=cfg['seed'],
        sample_fraction=1.0, do_crop=False,
    )
    """

    # --- metrics meter initialization ---
    meters = {
        'mad_full': AvgMeter(),
        'sad_unknown': AvgMeter(),
        'mse': AvgMeter(),
        'grad': AvgMeter(),
        'conn': AvgMeter(),
    }

    # --- main loop ---
    SHALLOW_WARMUP_FREEZE_EPOCHS = 2   # warmup (freeze post head)
    SHALLOW_WARMUP_LOWLR_EPOCHS = 4   # low LR

    for epoch in range(start_epoch, cfg['num_epochs']):
        print(f"[epoch {epoch}] enter loop", flush=True)

        if epoch == SHALLOW_WARMUP_FREEZE_EPOCHS:
            for p in post_params:
                p.requires_grad_(True)
            for g in optimizer.param_groups:
                if g.get("name") == "post":
                    g["lr"] = cfg["lr"] * 0.1
            print(f"[lr schedule] epoch {epoch}: unfreeze post, lr -> {cfg['lr']*0.1:.1e}")

        if epoch == SHALLOW_WARMUP_LOWLR_EPOCHS:
            for g in optimizer.param_groups:
                if g.get("name") == "post":
                    g["lr"] = cfg["lr"]
            print(f"[lr schedule] epoch {epoch}: post lr -> {cfg['lr']:.1e}")

        train_one_epoch(epoch, low_loader)
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


