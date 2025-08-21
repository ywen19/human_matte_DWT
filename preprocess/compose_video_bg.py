"""
Compose FG and BG with alpha blending;
Before blending, color harmonization based on reihard transfer is applied to FG
"""

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
import os
import csv
import gc
import shutil


# Global variable for per-process background pool
def init_pool(bgs):
    global background_images
    background_images = bgs


def check_ffmpeg():
    try:
        res = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except:
        return False


def composite_foreground_background(fg, alpha, bg):
    """Alpha-blend FG over BG."""
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_3 = np.repeat(alpha_f[:, :, None], 3, axis=2)
    fg_f = fg.astype(np.float32)
    bg_resized = cv2.resize(bg, (fg.shape[1], fg.shape[0])).astype(np.float32)
    comp = fg_f * alpha_3 + bg_resized * (1 - alpha_3)
    return comp.astype(np.uint8)


def reinhard_transfer_conservative(fg_lab, bg_lab, mask_ref, apply_mask):
    """Conservative Reinhard transfer: stats from mask_ref, apply only on apply_mask."""
    lab = fg_lab.copy().astype(np.float32)
    # If mask for stats is empty, skip transfer
    if not np.any(mask_ref):
        return fg_lab
    fg_vals = lab[mask_ref]
    bg_vals = bg_lab[mask_ref]
    # If either slice empty, skip transfer
    if fg_vals.size == 0 or bg_vals.size == 0:
        return fg_lab
    mu_s, std_s = fg_vals.mean(axis=0), fg_vals.std(axis=0)
    mu_t, std_t = bg_vals.mean(axis=0), bg_vals.std(axis=0)

    # Avoid division by zero
    std_s = np.where(std_s < 1e-6, 1.0, std_s)
    scales = np.clip(std_t / std_s, [0.8, 0.95, 0.95], [1.2, 1.02, 1.02])
    shifts = mu_t - mu_s * scales
    shifts = np.clip(shifts, [-10, -3, -3], [10, 3, 3])

    out_lab = lab.copy()
    # Apply only on masked area
    for c in range(3):
        ch = out_lab[..., c]
        if np.any(apply_mask):
            vals = ch[apply_mask] * scales[c] + shifts[c]
            ch[apply_mask] = vals
            out_lab[..., c] = ch
    return np.clip(out_lab, 0, 255).astype(np.uint8)


def process_video_pair(args):
    fgr_path, pha_path, out_fgr, out_pha, out_hmfg, split_txt = args
    print(f"[Worker] Processing {fgr_path.name} in {split_txt}")

    # choose and load background
    bg_path = random.choice(background_images)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        print("Failed to load background")
        return None

    cap_fg = cv2.VideoCapture(str(fgr_path))
    cap_alpha = cv2.VideoCapture(str(pha_path))
    if not cap_fg.isOpened() or not cap_alpha.isOpened():
        print("Cannot open videos")
        return None

    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    cnt = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or cnt == 0:
        print("Invalid video")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_fg = Path(tmpdir)/"fg_comp"
        dir_pha = Path(tmpdir)/"pha"
        dir_hm = Path(tmpdir)/"fg_hm"
        for d in (dir_fg, dir_pha, dir_hm): d.mkdir()

        # read first frame for stats
        ret0, frame0 = cap_fg.read()
        _, pha0 = cap_alpha.read()
        if not ret0:
            print("No frames")
            return None
        alpha0 = cv2.cvtColor(pha0, cv2.COLOR_BGR2GRAY)
        pure0 = (frame0.astype(np.float32) * (alpha0.astype(np.float32)/255)[...,None]).astype(np.uint8)
        bg0 = cv2.resize(bg, (frame0.shape[1], frame0.shape[0]))
        pure0_lab = cv2.cvtColor(pure0, cv2.COLOR_BGR2LAB)
        bg0_lab = cv2.cvtColor(bg0, cv2.COLOR_BGR2LAB)
        mask = alpha0 > 0

        def transfer_frame(frm, alpha):
            # norm alpha
            alpha_f = alpha.astype(np.float32) / 255.0

            # premultiplied -> unpremultiplied
            premult = frm.astype(np.float32) * alpha_f[...,None]
            unpremult = premult / np.maximum(alpha_f[...,None], 1e-6)

            # to Lab space then reihard color transfer
            lab = cv2.cvtColor(unpremult.astype(np.uint8), cv2.COLOR_BGR2LAB)
            lab_tr = reinhard_transfer_conservative(lab, bg0_lab, mask, alpha_f>0)

            # back to RGB
            return cv2.cvtColor(lab_tr, cv2.COLOR_LAB2BGR)

        cap_fg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_alpha.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret_fg, frm = cap_fg.read()
            ret_ph, pha = cap_alpha.read()
            if not ret_fg or not ret_ph:
                break
            alpha = cv2.cvtColor(pha, cv2.COLOR_BGR2GRAY)
            hm = transfer_frame(frm, alpha)
            comp = composite_foreground_background(hm, alpha, bg)
            cv2.imwrite(str(dir_fg/f"f_{idx:05d}.png"), comp)
            cv2.imwrite(str(dir_pha/f"a_{idx:05d}.png"), cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(str(dir_hm/f"h_{idx:05d}.png"), hm)
            # free memory for this iteration
            del hm, comp, alpha, frm, pha
            gc.collect()
            idx += 1

        cap_fg.release()
        cap_alpha.release()
        if idx == 0:
            print("No frames processed")
            return None

        def encode_seq(pattern, out_path):
            tmp = str(out_path) + ".tmp.mp4"
            cmd = [
                'ffmpeg', '-y', '-framerate', str(int(fps)), '-i',
                str(Path(tmpdir)/pattern), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', tmp
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            Path(tmp).rename(out_path)

        try:
            encode_seq("fg_comp/f_%05d.png", out_fgr)
            encode_seq("pha/a_%05d.png", out_pha)
            encode_seq("fg_hm/h_%05d.png", out_hmfg)
        except subprocess.CalledProcessError as e:
            print("FFmpeg error", e)
            # cleanup temp files on failure
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None

        print(f"Done ({idx} frames)")
        # final garbage collection
        gc.collect()
        return (fgr_path.name, str(bg_path), split_txt)


if __name__ == "__main__":
    if not check_ffmpeg():
        print("FFmpeg not found")
        exit(1)

    root = Path("../data/VideoMatte240K")
    bg_dir = root/"Backgrounds"
    out_root = Path("../data/video_composed")
    bgs = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    assert bgs, "No backgrounds found"

    # prepare outputs and resume files
    splits = ["train", "test"]
    resume_files = {}
    for split in splits:
        for sub in ("fgr", "pha", "harmonized_fg"):
            (out_root/split/sub).mkdir(parents=True, exist_ok=True)
        txt_path = out_root/f"{split}_processed.txt"
        if not txt_path.exists():
            txt_path.write_text("")
        processed = set(txt_path.read_text().splitlines())
        resume_files[split] = (txt_path, processed)

    # build jobs, skipping processed
    jobs = []
    for split in splits:
        fgs = sorted((root/split/"fgr").glob("*.mp4"))
        phs = sorted((root/split/"pha").glob("*.mp4"))
        txt_path, processed = resume_files[split]
        for fg, ph in zip(fgs, phs):
            """if split == "train":
                if int(fg.stem) <151:
                    continue"""
            if fg.name in processed:
                continue
            jobs.append((
                fg, ph,
                out_root/split/"fgr"/f"{fg.stem}.mp4",
                out_root/split/"pha"/f"{ph.stem}.mp4",
                out_root/split/"harmonized_fg"/f"{fg.stem}.mp4",
                split
            ))

    mapping_csv = out_root/'background_mapping.csv'
    if not mapping_csv.exists():
        with open(mapping_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_name', 'background_path'])

    nproc = max(1, min(cpu_count()//2, 16))
    print(f"Using {nproc} processes")
    with Pool(nproc, initializer=init_pool, initargs=(bgs,)) as pool:
        for res in tqdm(pool.imap_unordered(process_video_pair, jobs), total=len(jobs)):
            if res:
                video_name, bg_used, split = res
                rel_bg = os.path.relpath(bg_used, start=str(out_root.parent))
                with open(mapping_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_name, rel_bg])
                txt_path, _ = resume_files[split]
                with open(txt_path, 'a') as f:
                    f.write(video_name + '\n')

    print(f"Mapping CSV written to {mapping_csv}")
    print("Complete!")
