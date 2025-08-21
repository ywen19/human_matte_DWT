import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures
import argparse
import multiprocessing
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames and clipped videos from defocused videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing train/test datasets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument(
        '--dataset', type=str, default='both', choices=['train', 'test', 'both'], help='Dataset to process'
    )
    parser.add_argument('--ratio', type=float, default=0.2, help='Sampling ratio (e.g., 0.1 for 10%)')
    return parser.parse_args()


# random sample frames by the given ratio
def sample_continuous_segment(total_frames, ratio):
    segment_length = max(1, int(ratio * total_frames))
    if segment_length >= total_frames:
        return list(range(total_frames))
    start_idx = random.randint(0, total_frames - segment_length)
    return list(range(start_idx, start_idx + segment_length))


# save sampled(clipped) video
def save_clipped_video(video_path, output_video_dir, video_id, frame_ids):
    os.makedirs(output_video_dir, exist_ok=True)
    output_path = os.path.join(output_video_dir, f"{video_id}.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_ids_set = set(frame_ids)
    read_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx in frame_ids_set:
            out.write(frame)
        read_idx += 1

    cap.release()
    out.release()


# for each video, extract all frames as images
def process_video_cpu(video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir, ratio):
    try:
        # amount of frames
        cap = cv2.VideoCapture(fgr_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # frames to be sampled out
        frame_ids = sample_continuous_segment(total_frames, ratio)

        # prepare data directories to output sampled data
        video_fgr_root = os.path.join(output_fgr_dir, video_id)
        video_pha_root = os.path.join(output_pha_dir, video_id)
        fgr_frames_dir = os.path.join(video_fgr_root, "frames")
        pha_frames_dir = os.path.join(video_pha_root, "frames")
        fgr_video_dir = os.path.join(video_fgr_root, "video")
        pha_video_dir = os.path.join(video_pha_root, "video")
        os.makedirs(fgr_frames_dir, exist_ok=True)
        os.makedirs(pha_frames_dir, exist_ok=True)

        # get FG
        fgr_cap = cv2.VideoCapture(fgr_path)
        for idx in frame_ids:
            fgr_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = fgr_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx_str = f"{idx:04d}"
                im = Image.fromarray(frame_rgb)
                im.save(
                    os.path.join(fgr_frames_dir, f"{idx_str}.png"),
                    optimize=True,
                    compress_level=9)
        fgr_cap.release()

        # get alpha
        pha_cap = cv2.VideoCapture(pha_path)
        for idx in frame_ids:
            pha_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = pha_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                idx_str = f"{idx:04d}"
                im = Image.fromarray(frame_rgb, mode='L')
                im.save(
                    os.path.join(pha_frames_dir, f"{idx_str}.png"),
                    optimize=True,
                    compress_level=9)
        pha_cap.release()

        # save sampled(clipped FG and alpha videos)
        save_clipped_video(fgr_path, fgr_video_dir, video_id, frame_ids)
        save_clipped_video(pha_path, pha_video_dir, video_id, frame_ids)

        return f"Processed {video_id}"
    except Exception as e:
        return f"Error processing {video_id}: {e}"


# wrapper to prevent lambda pickling issues in multiprocessing
def process_video_cpu_wrapper(args):
    return process_video_cpu(*args)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    args = parse_args()

    print("Configuration:")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling ratio: {args.ratio}")

    # subsets we want to process
    datasets = []
    if args.dataset in ('train', 'both'):
        datasets.append('train')
    if args.dataset in ('test', 'both'):
        datasets.append('test')

    for dataset in datasets:
        input_fgr_dir = os.path.join(args.input_dir, dataset, "fgr")
        input_pha_dir = os.path.join(args.input_dir, dataset, "pha")
        output_fgr_dir = os.path.join(args.output_dir, dataset, "fgr")
        output_pha_dir = os.path.join(args.output_dir, dataset, "alpha")

        os.makedirs(output_fgr_dir, exist_ok=True)
        os.makedirs(output_pha_dir, exist_ok=True)

        # —— load processed list for break-point resuming —— #
        processed_txt = os.path.join(args.output_dir, f"{dataset}_processed.txt")
        if not os.path.exists(processed_txt):
            open(processed_txt, 'w').close()
        with open(processed_txt, 'r') as f:
            processed_set = set(line.strip() for line in f if line.strip())

        # find all video files to be processed
        video_files = [
            f for f in os.listdir(input_fgr_dir)
            if os.path.isfile(os.path.join(input_fgr_dir, f))
               and Path(f).suffix.lower() in ['.mp4', '.avi', '.mov']
        ]
        print(f"Found {len(video_files)} {dataset} videos")

        # build task list and skip already processed ones
        tasks = []
        for vf in video_files:
            vid = Path(vf).stem
            if vid in processed_set:
                continue
            fgr_path = os.path.join(input_fgr_dir, vf)
            pha_path = os.path.join(input_pha_dir, vf)
            if not os.path.exists(pha_path):
                print(f"Missing alpha file {pha_path}, skipping {vid}")
                continue
            tasks.append((vid, fgr_path, pha_path, output_fgr_dir, output_pha_dir, args.ratio))

        # implement real-time writing with ProcessPoolExecutor and as_completed
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(multiprocessing.cpu_count(), 32)) as executor:
            futures = {executor.submit(process_video_cpu_wrapper, t): t[0] for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {dataset}"):
                vid = futures[future]
                res = future.result()
                print(res)

                # —— real-time appending to processed list —— #
                if res.startswith("Processed"):
                    with open(processed_txt, 'a') as f:
                        f.write(vid + "\n")

    print("Complete!")
