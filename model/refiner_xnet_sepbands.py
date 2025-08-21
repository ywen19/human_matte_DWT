import os
# Bypass nvidia-modprobe errors by redirecting to no-op
os.environ['NVIDIA_MODPROBE'] = '/bin/true'

import cv2
import random
import torch
import kornia
import numpy as np
import gc
from pathlib import Path

# Hard-coded paths
ROOT = Path('../data/VideoMatte240K')
BG_DIR = ROOT / 'Backgrounds'
OUT_ROOT = Path('../data/video_composed_gpu')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Utility functions
def to_tensor_cuda(img_uint8):
    # Convert and move to GPU
    img = torch.from_numpy(img_uint8).float().permute(2,0,1) / 255.0
    return img.cuda()

def to_numpy_uint8(tensor):
    img = tensor.clamp(0.0,1.0).cpu().permute(1,2,0).numpy()
    return (img * 255.0).astype(np.uint8)

# Reinhard transfer
def reinhard_transfer_torch(pure_lab, bg_lab, mask):
    channels, H, W = pure_lab.shape
    mask_flat = mask.view(-1)
    lab_flat = pure_lab.reshape(channels, -1).permute(1,0)
    bg_flat  = bg_lab.reshape(channels, -1).permute(1,0)
    src_vals, tgt_vals = lab_flat[mask_flat], bg_flat[mask_flat]
    mu_s, std_s = src_vals.mean(0), src_vals.std(0, unbiased=False)
    mu_t, std_t = tgt_vals.mean(0), tgt_vals.std(0, unbiased=False)
    scales = (std_t / (std_s + 1e-6)).clamp(min=0.8, max=1.2)
    shifts = (mu_t - mu_s * scales).clamp(min=-10.0/255.0, max=10.0/255.0)
    out = pure_lab.clone()
    for c in range(channels):
        ch = out[c].reshape(-1)
        ch[mask_flat] = ch[mask_flat] * scales[c] + shifts[c]
        out[c] = ch.reshape(H, W)
    return out

# Process a single pair
def process_pair(fg_path, pha_path, bg_list):
    bg_path = random.choice(bg_list)
    bg_bgr = cv2.imread(str(bg_path))
    cap_fg = cv2.VideoCapture(str(fg_path))
    cap_pha = cv2.VideoCapture(str(pha_path))
    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    cnt_fg, cnt_ph = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_pha.get(cv2.CAP_PROP_FRAME_COUNT))
    if not (cap_fg.isOpened() and cap_pha.isOpened()) or fps <= 0 or cnt_fg == 0 or cnt_ph == 0:
        cap_fg.release(); cap_pha.release()
        return False
    w, h = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_comp = OUT_ROOT / fg_path.name
    vw = cv2.VideoWriter(str(out_comp), fourcc, fps, (w,h))
    # First-frame stats
    ret_f0, f0 = cap_fg.read(); ret_p0, p0 = cap_pha.read()
    if not ret_f0 or not ret_p0:
        cap_fg.release(); cap_pha.release(); vw.release()
        return False
    a0 = cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY)
    a0f = torch.from_numpy(a0.astype(np.float32)/255.0).cuda()
    mask0 = a0f > 0.0
    fg0 = to_tensor_cuda(f0); bg0 = to_tensor_cuda(cv2.resize(bg_bgr,(w,h)))
    pure0 = fg0 * a0f.unsqueeze(0)
    pure0_lab = kornia.color.rgb_to_lab(pure0.unsqueeze(0))[0]
    bg0_lab = kornia.color.rgb_to_lab(bg0.unsqueeze(0))[0]
    cap_fg.set(cv2.CAP_PROP_POS_FRAMES, 0); cap_pha.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Loop
    while True:
        ret_f, f = cap_fg.read(); ret_p, p = cap_pha.read()
        if not ret_f or not ret_p: break
        a = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        af = torch.from_numpy(a.astype(np.float32)/255.0).cuda()
        fg_t = to_tensor_cuda(f); bg_t = to_tensor_cuda(cv2.resize(bg_bgr,(w,h)))
        pure = fg_t * af.unsqueeze(0)
        lab = kornia.color.rgb_to_lab(pure.unsqueeze(0))[0]
        tr_lab = reinhard_transfer_torch(lab, bg0_lab, mask0)
        tr_rgb = kornia.color.lab_to_rgb(tr_lab.unsqueeze(0))[0]
        hm = tr_rgb * af.unsqueeze(0)
        comp = hm + bg_t * (1.0 - af.unsqueeze(0))
        vw.write(to_numpy_uint8(comp))
        # cleanup
        del fg_t, bg_t, pure, lab, tr_lab, tr_rgb, hm, comp
        torch.cuda.empty_cache(); gc.collect()
    cap_fg.release(); cap_pha.release(); vw.release()
    del fg0, bg0, pure0, pure0_lab, bg0_lab, a0f, mask0
    torch.cuda.empty_cache(); gc.collect()
    return True

if __name__ == '__main__':
    bg_list = list(BG_DIR.glob('*.*'))
    for split in ['train','test']:
        for fg in (ROOT/split/'fgr').glob('*.mp4'):
            pha = ROOT/split/'pha'/fg.name
            print(f"[INFO] {split}/{fg.name}")
            result = process_pair(fg, pha, bg_list)
            print('[OK]' if result else '[FAIL]')
