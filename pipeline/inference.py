#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for refiner_xnet (rgbembed)
"""

import argparse
import os
import sys
import glob
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.refiner_xnet_rgbembed import refiner_xnet
from pipeline.utils import *

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.uint8)


def to_tensor(
        rgb: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = rgb.shape[:2]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # [3,H,W]
    t = t.unsqueeze(0)                                         # [1,3,H,W]
    t = (t - IMAGENET_MEAN) / IMAGENET_STD                     # [1,3,H,W]
    t = t.to(device)
    return t, (h, w)


def pad_to_multiple(
        t: torch.Tensor, multiple: int = 32, mode: str = 'reflect'
) -> tuple[torch.Tensor, tuple[int,int,int,int]]:
    _, _, H, W = t.shape
    newH = (H + multiple - 1) // multiple * multiple
    newW = (W + multiple - 1) // multiple * multiple
    pad_b = newH - H
    pad_r = newW - W
    if pad_b or pad_r:
        t = F.pad(t, (0, pad_r, 0, pad_b), mode=mode)
    return t, (0, pad_r, 0, pad_b)


def crop_like(t: torch.Tensor, hw: tuple[int,int]) -> torch.Tensor:
    H, W = hw
    return t[..., :H, :W]


def build_model(device: torch.device,
                wavelet_list: list[str] = ('db1','db1'),
                base_channels: int = 64,
                num_layers: int = 4,
                blocks_per_layer: int = 3,
                d_head: int = 128) -> nn.Module:
    model = refiner_xnet(
        wavelet_list=list(wavelet_list),
        base_channels=base_channels,
        num_layers=num_layers,
        blocks_per_layer=blocks_per_layer,
        d_head=d_head,
        modes_bn={'row','col','diag','global'},
        modes_mid={'row','col','diag','global'},
    )
    bn_to_gn(model)
    model.eval().to(device)
    return model


def load_checkpoint(
        model: nn.Module, ckpt_path: str, device: torch.device
) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"couldn't find checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[load_state_dict] missing weights: {len(missing)}，e.g.: {missing[:5]}"
        )
    if unexpected:
        print(
            f"[load_state_dict] extra weights: {len(unexpected)}，e.g.: {unexpected[:5]}"
        )


@torch.no_grad()
def infer_one(model: nn.Module, rgb_np: np.ndarray, device: torch.device,
              pad_multiple: int = 32) -> np.ndarray:
    # [1,3,H,W]
    rgb_t, hw = to_tensor(rgb_np, device)
    rgb_t, _  = pad_to_multiple(rgb_t, multiple=pad_multiple, mode='reflect')

    # have to pass in dummy mask
    # but actually not using for our model
    # our model is guide-free
    dummy = torch.zeros(
        rgb_t.shape[0], 1, rgb_t.shape[-2], rgb_t.shape[-1],
        device=rgb_t.device, dtype=rgb_t.dtype
    )

    outs = model(rgb_t, dummy)
    if isinstance(outs, (list, tuple)):
        out = outs[-1]
    elif isinstance(outs, dict):
        for k in ('out', 'pred', 'alpha', 'mask', 'last', 'final'):
            if k in outs:
                out = outs[k]
                break
        else:
            out = list(outs.values())[-1]
    else:
        out = outs

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"model return not tensor，but {type(out)}")

    # match shapes
    if out.ndim == 5:
        # [B, T, C, H, W]，collapse T
        out = out[:, -1, ...]
    elif out.ndim == 3:
        # [C, H, W]，add B dim
        out = out.unsqueeze(0)
    elif out.ndim == 2:
        # [H, W]，add color channel and B dim
        out = out.unsqueeze(0).unsqueeze(0)

    if out.ndim != 4:
        raise RuntimeError(
            f"Expected our 4 dim [B,C,H,W], but {out.ndim} 维，shape={tuple(out.shape)}"
        )

    # back to original size if cropped
    recon = crop_like(out, hw).clamp(0, 1)
    if recon.shape[1] < 1:
        raise RuntimeError(f"out channel not enough：{recon.shape}")

    alpha = (recon[0, 0].float().cpu().numpy() * 255.0).round().astype(np.uint8)
    return alpha


def save_alpha(alpha: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(alpha, mode='L').save(out_path)


def save_overlay(rgb: np.ndarray, alpha: np.ndarray, out_path: str, bg_color=(121,255,155)):
    """save detected FG to green BG
    bg_color: default green (0,255,0)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fg = rgb.astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    bg = np.array(bg_color, dtype=np.float32)
    comp = fg * a + bg * (1.0 - a)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    Image.fromarray(comp).save(out_path)


def list_images(path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files: List[str] = []

    if os.path.isfile(path):
        # images under the path (supports 1 or multiple)
        if path.lower().endswith(exts):
            files = [path]
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    files.append(os.path.join(root, fname))
    else:
        for f in glob.glob(path):
            if f.lower().endswith(exts) and os.path.isfile(f):
                files.append(f)
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="refiner_xnet (rgbembed) inference")
    parser.add_argument(
        '--image', default='../data/video_composed_frames/train/fgr/0015/frames/'
    )
    parser.add_argument(
        '--mask', default='../data/video_composed_frames/train/fgr/0015/mask/0137.png'
    )
    parser.add_argument(
        '--checkpoint', default='./checkpoints_xnet_db1_final_plateau/epoch33.pth'
    )
    parser.add_argument('--outdir', default='../outputs')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--pad-multiple', type=int, default=32)
    parser.add_argument('--save-overlay', action='store_true')
    parser.add_argument('--wavelets', nargs='+', default=['db1','db1'])
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--blocks-per-layer', type=int, default=3)
    parser.add_argument('--d-head', type=int, default=128)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = build_model(device,
                        wavelet_list=args.wavelets,
                        base_channels=args.base_channels,
                        num_layers=args.num_layers,
                        blocks_per_layer=args.blocks_per_layer,
                        d_head=args.d_head)
    load_checkpoint(model, args.checkpoint, device)
    print('model loaded')

    images = list_images(args.image)
    if not images:
        raise FileNotFoundError(f"Couldn't find image：{args.image}")

    for img_path in images:
        try:
            rgb_np = load_rgb(img_path)
            rgb_np = np.array(Image.fromarray(rgb_np).resize((1280, 720), Image.BILINEAR))
            alpha = infer_one(model, rgb_np, device,
                              pad_multiple=args.pad_multiple)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_alpha = os.path.join(args.outdir, base + args.suffix + '.png')
            save_alpha(alpha, out_alpha)
            print(args.save_overlay)
            if args.save_overlay:
                out_overlay = os.path.join(args.outdir, 'overlay_' + base + '.png')
                save_overlay(rgb_np, alpha, out_overlay)
            print(f"[OK] {img_path} -> {out_alpha}")
        except Exception as e:
            print(f"[ERR] {img_path}: {e}")


if __name__ == '__main__':
    main()