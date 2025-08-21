#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final YOLO+SAM2 batch mask generator
multimask_output=True + masks[1] + process bboxes one-by-one
bbox filtering: drop small persons in the background
for filtering, no constraints on person aspect ratio, only consider overall size/area ratio
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import time

import sys
sys.path.append(os.path.abspath("../sam2"))
from ultralytics import YOLO

# SAM2导入
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pathlib import Path
import multiprocessing
import signal
import subprocess

# —— nvidia-modprobe ——
import subprocess as _subproc

_orig_run = _subproc.run
_orig_call = _subproc.call


def _patch_args(args):
    if isinstance(args, (list, tuple)) and any('nvidia-modprobe' in x for x in args):
        return [x for x in args if x != '-s']
    return args


def _run(*p, **kw):
    p = list(p)
    p[0] = _patch_args(p[0])
    return _orig_run(*p, **kw)


def _call(*p, **kw):
    p = list(p)
    p[0] = _patch_args(p[0])
    return _orig_call(*p, **kw)


_subproc.run = _run
_subproc.call = _call

# ---------- config ----------
YOLO_MODEL_PATH = "yolov8n.pt"
MAX_LOGICAL_WORKERS = 4

# filter config
GLOBAL_MIN_PERSON_SIZE = 0.01
GLOBAL_MIN_CONFIDENCE = 0.3
GLOBAL_DEBUG_FILTERING = False


def load_completed_log(split):
    """load completed video list"""
    path = f"completed_masks_method1_{split}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(f.read().splitlines())
    return set()


def log_completed(split, video_id):
    """log finished videos"""
    path = f"completed_masks_method1_{split}.txt"
    with open(path, "a") as f:
        f.write(video_id + "\n")


def fill_small_holes_opencv(mask, max_area_ratio=0.005):
    """important post-processing: fill small holes"""
    h, w = mask.shape
    img_area = h * w
    bin_mask = (mask == 255).astype(np.uint8)
    inv_img = cv2.bitwise_not(bin_mask * 255)
    flood = inv_img.copy()
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 0)
    internal = (flood == 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(internal)
    for lid in range(1, num_labels):
        region = (labels == lid)
        if np.sum(region) <= max_area_ratio * img_area:
            bin_mask[region] = 1
    return (bin_mask * 255).astype(np.uint8)


def save_mask(mask_array, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(mask_array.astype(np.uint8), mode='L').save(
        output_path,
        optimize=True,
        compress_level=9
    )
    return os.path.exists(output_path)


def filter_person_detections(boxes, w, h, frame_path=""):
    """
    remove small background persons from detected bboxes based on size ratio
    """
    global GLOBAL_MIN_PERSON_SIZE, GLOBAL_MIN_CONFIDENCE, GLOBAL_DEBUG_FILTERING
    
    person_detections = []
    valid_boxes_info = []  # for debug
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        confidence = boxes.conf[i].cpu().numpy()
        class_id = int(boxes.cls[i].cpu().numpy())
        class_name = yolo_model.names[class_id]
        
        # handle person-class detections only
        if class_name == 'person':
            # calculate the dimensions of the bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # calculate size relative to the image dimensions
            relative_width = bbox_width / w
            relative_height = bbox_height / h
            relative_area = bbox_area / (w * h)
            
            # filter condition
            # 1. minimum size filtering - remove small persons in the background
            min_relative_width = 0.1   # require bbox width ≥ 5% of image width
            min_relative_height = 0.15   # require bbox height ≥ 10% of image width
            min_relative_area = GLOBAL_MIN_PERSON_SIZE
            
            # 2. confidence
            min_confidence = GLOBAL_MIN_CONFIDENCE
            
            # apply filter condition
            size_valid = (relative_width >= min_relative_width and 
                         relative_height >= min_relative_height and 
                         relative_area >= min_relative_area)
            
            confidence_valid = confidence >= min_confidence
            
            # collect valid info
            valid_boxes_info.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'size': f"{relative_width:.3f}x{relative_height:.3f} ({relative_area:.4f})",
                'size_valid': size_valid,
                'conf_valid': confidence_valid,
                'overall_valid': size_valid and confidence_valid
            })

            if size_valid and confidence_valid:
                person_detections.append([x1, y1, x2, y2])
    
    # debug output - show filtering details (only in debug mode)
    if len(valid_boxes_info) > 0 and GLOBAL_DEBUG_FILTERING:
        frame_name = frame_path.split('/')[-1] if frame_path else "unknown"
        print(f"[DEBUG] 检测框过滤详情 ({frame_name}):")
        for i, info in enumerate(valid_boxes_info):
            status = "✓保留" if info['overall_valid'] else "✗过滤"
            filter_reasons = []
            if not info['size_valid']:
                filter_reasons.append('尺寸过小')
            if not info['conf_valid']:
                filter_reasons.append('置信度低')
            
            reason = '通过' if info['overall_valid'] else '、'.join(filter_reasons)
            
            print(f"  人物{i+1}: {status} | 置信度:{info['confidence']:.3f} | "
                  f"相对尺寸:{info['size']} | 长宽比:{info['aspect']} | "
                  f"过滤原因: {reason}")
        print(f"  最终保留: {len(person_detections)}/{len(valid_boxes_info)} 个检测框")
    
    return person_detections


def init_models():
    """per-worker initialization of YOLO and SAM2 model"""
    global yolo_model, sam_predictor
    
    print("Worker模型初始化...")
    
    # load YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO loaded")
    
    # load sam2
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    config_dir = os.path.abspath("../sam2/sam2/configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sam2_hiera_l.yaml")
        OmegaConf.resolve(cfg)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = instantiate(cfg.model, _recursive_=True)
        
        # load sam2 weights
        model_path = "../local_sam2_hiera_large/sam2_hiera_large.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        
        model = model.to(device).eval()
        sam_predictor = SAM2ImagePredictor(model)
        print("sam2 loaded")


@torch.inference_mode()
@torch.autocast("cuda")
def generate_mask_for_frame(frame_path):
    """
    generate masks for a single frame
    - use multimask_output=True and always select masks[1]
    - process bboxes one-by-one (no batching)
    - add bbox filtering (no aspect ratio constraint)
    - use YOLO bboxes as SAM2 prompts
    """
    try:
        # load image
        image = Image.open(frame_path).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Yolo person detect
        results = yolo_model(image_np, verbose=False)
        
        if results[0].boxes is None:
            # return empty mask if no subject detected
            return np.zeros((h, w), dtype=np.uint8)
        
        boxes = results[0].boxes
        
        # filter
        person_detections = filter_person_detections(boxes, w, h, frame_path)
        
        if not person_detections:
            # return empty mask if all detected subjects are filtered out
            return np.zeros((h, w), dtype=np.uint8)
        
        # sam2 segment
        sam_predictor.set_image(image_np)
        all_person_masks = []
        
        for bbox in person_detections:
            # use yolo bbox as sam2 hint
            # bbox: [x1, y1, x2, y2]
            input_box = np.array(bbox, dtype=np.float32)
            
            try:
                masks, scores, logits = sam_predictor.predict(
                    box=input_box,
                    multimask_output=True,  # multiple candidate mask
                )
                
                # we run some tests, and use mask[1] yields more stable results
                if len(masks) > 1:
                    selected_mask = masks[1]  # pick medium-quality mask (typically best among tested)
                else:
                    selected_mask = masks[0]  # backup
                
                all_person_masks.append(selected_mask)
                
            except Exception as e:
                print(f"[WARNING] Fail to segment: {e}")
                continue
        
        if not all_person_masks:
            # return empty mask if all sam segment fails
            return np.zeros((h, w), dtype=np.uint8)
        
        # multi-person mask combination
        if len(all_person_masks) == 1:
            # if single person, use mask directly
            final_mask = all_person_masks[0]
        else:
            # if multiple persons, OR all masks together
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in all_person_masks:
                combined_mask = np.logical_or(combined_mask, mask)
            final_mask = combined_mask
        
        # post process
        final_mask_uint8 = (final_mask * 255).astype(np.uint8)
        
        # use morphological close to link nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask_uint8 = cv2.morphologyEx(final_mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # fill small holes
        final_mask_processed = fill_small_holes_opencv(final_mask_uint8)
        
        return final_mask_processed
        
    except Exception as e:
        print(f"[ERROR] Fail to process frame: {frame_path}: {e}")
        # return None if fail
        try:
            image = Image.open(frame_path).convert('RGB')
            image_np = np.array(image)
            h, w = image_np.shape[:2]
        except:
            h, w = 480, 640  # default size
        return np.zeros((h, w), dtype=np.uint8)


def process_single_video(args):
    """process all frames of a single video"""
    split, video_id, frame_dir, first_frame_only = args
    
    print(f"[INFO] Start process video: {video_id} (split: {split})")
    
    # get all frames
    frame_files = sorted(Path(frame_dir).glob("*.png"))
    if first_frame_only:
        frame_files = frame_files[:1]
    
    if not frame_files:
        print(f"[WARNING] Video {video_id} contains no .png files")
        return split, video_id
    
    # output directory
    output_dir = Path(frame_dir).parent / "mask"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    total_frames = len(frame_files)
    
    # for every frame
    for i, frame_path in enumerate(frame_files):
        try:
            # generate mask
            mask = generate_mask_for_frame(str(frame_path))

            if first_frame_only:
                output_name = "first_frame_mask.png"
            else:
                output_name = frame_path.name
            
            output_path = output_dir / output_name
            
            if save_mask(mask, str(output_path)):
                processed_count += 1
            else:
                print(f"[ERROR] Fail to save: {output_path}")
            
            # report progress
            if (i + 1) % 20 == 0 or i == total_frames - 1:
                progress = (i + 1) / total_frames * 100
                print(f"[PROGRESS] Video {video_id}: {i+1}/{total_frames} ({progress:.1f}%)")
                
        except Exception as e:
            print(f"[ERROR] Fail to process frame {frame_path}: {e}")
    
    success_rate = processed_count / total_frames * 100
    print(f"[INFO] Video {video_id} complete on: {processed_count}/{total_frames} Frame ({success_rate:.1f}%)")
    
    return split, video_id


# ---------- batch process ----------
def estimate_safe_workers(mem_per_worker_mb=3500, safety_margin=0.85, min_workers=1):
    """estimate safe number of workers based on GPU memory"""
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            reserved_mem = torch.cuda.memory_reserved(0) / (1024**2)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**2)
            
            available_mem = (total_mem - max(reserved_mem, allocated_mem)) * safety_margin
            estimated_workers = int(available_mem // mem_per_worker_mb)
            
            return max(min_workers, min(MAX_LOGICAL_WORKERS, estimated_workers))
        else:
            return max(min_workers, min(MAX_LOGICAL_WORKERS, os.cpu_count() // 2))
    except:
        return max(min_workers, min(MAX_LOGICAL_WORKERS, 2))


def process_dataset(root_dir, first_frame_only=False, max_workers=None):
    """process whole dataset"""
    print("start YOLO+SAM2 batch mask generator")
    print("multimask_output=True + masks[1]")
    print("per bbox")
    print("="*70)
    
    # scan videos to be processed
    video_tasks = []
    total_videos_found = 0
    
    for split in ["train", "test"]:
        split_dir = Path(root_dir) / split / "fgr"
        if not split_dir.exists():
            print(f"[WARNING] path not exists: {split_dir}")
            continue
        
        # load finished video list
        completed_videos = load_completed_log(split)
        all_videos = sorted([d for d in os.listdir(split_dir) if os.path.isdir(split_dir / d)])
        pending_videos = [v for v in all_videos if v not in completed_videos]
        
        total_videos_found += len(all_videos)
        
        print(f"{split.upper()} split:")
        print(f"overall video numbers: {len(all_videos)}")
        print(f"Complete: {len(completed_videos)}")
        print(f"TBD: {len(pending_videos)}")
        
        # add to task
        for video_id in pending_videos:
            frame_dir = split_dir / video_id / "frames"
            if frame_dir.exists():
                video_tasks.append((split, video_id, str(frame_dir), first_frame_only))
            else:
                print(f"[WARNING] Frame path not exists: {frame_dir}")
    
    if not video_tasks:
        print("All videos processed!")
        return
    
    # estimate worker amount
    if max_workers:
        workers = max_workers
        print(f"set worker count manually: {workers}")
    else:
        workers = estimate_safe_workers()
        print(f"auto estimate worker count: {workers}")
    
    print(f"process config:")
    print(f"data path: {root_dir}")
    print(f"mode: {'first frame' if first_frame_only else 'all frames'}")
    print(f"video to be processed: {len(video_tasks)} 个")
    print(f"worker amount: {workers}")
    print(f"segment: YOLO bbox hint + masks[1] + per bbox")
    print(f"filter condition: min area={GLOBAL_MIN_PERSON_SIZE:.3f}, min confidence={GLOBAL_MIN_CONFIDENCE:.2f}")
    print(f"debug mode: {'Enable' if GLOBAL_DEBUG_FILTERING else 'Disable'}")
    
    # start batch process
    print("\n start batch process ..")
    start_time = time.time()
    
    ctx = multiprocessing.get_context("spawn")
    completed_count = 0
    
    with ctx.Pool(processes=workers, initializer=init_models) as pool:
        try:
            for split, video_id in pool.imap_unordered(process_single_video, video_tasks):
                # record complete status
                log_completed(split, video_id)
                completed_count += 1
                
                # progress and ETA remained
                elapsed_time = time.time() - start_time
                progress = completed_count / len(video_tasks) * 100
                
                if completed_count > 0:
                    avg_time_per_video = elapsed_time / completed_count
                    eta_seconds = avg_time_per_video * (len(video_tasks) - completed_count)
                    eta_minutes = eta_seconds / 60
                else:
                    eta_minutes = 0
                
                print(f"[COMPLETED] {video_id} (split: {split}) "
                      f"[{completed_count}/{len(video_tasks)}, {progress:.1f}%, ETA: {eta_minutes:.1f}min]")
                
        except KeyboardInterrupt:
            print("\n Get break signal, stop process safely ..")
            pool.terminate()
            pool.join()
            print(f"Stopped safely, completed {completed_count}/{len(video_tasks)} videos")
            return
    
    # 处理完成统计
    total_time = time.time() - start_time
    print(f"\nBatch process finished")
    print(f"Process statics:")
    print(f"Overall time: {total_time/60:.1f} min")
    print(f"Number of videos: {completed_count}")
    if completed_count > 0:
        print(f"Avg speed: {total_time/completed_count:.1f} sec/video")
    print(f"multimask_output=True + masks[1] + per bbox")


def kill_orphan_processes():
    """cleanup orphaned processes if any"""
    try:
        import psutil
        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)
        
        for child in current_process.children(recursive=True):
            try:
                child.terminate()
            except:
                pass
    except ImportError:
        # fallback: simple method since psutil is not available
        try:
            subprocess.run("pkill -f yolo_sam2_mask_extract", shell=True, check=False)
        except:
            pass


if __name__ == "__main__":
    # multiprocess setup
    multiprocessing.set_start_method("spawn", force=True)
    
    # kill orphan
    kill_orphan_processes()
    
    # command line args
    import argparse
    parser = argparse.ArgumentParser(
        description="YOLO+SAM2(multimask_output=True + per bbox)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Usage:
              python yolo_sam2_mask_extract.py --data_dir /path/to/data --mode first_frame
              python yolo_sam2_mask_extract.py --data_dir /path/to/data --mode all_frames --workers 4
              python yolo_sam2_mask_extract.py --data_dir /path/to/data --mode first_frame --debug_filtering --min_person_size 0.02
            
            Features:
              - Fallback to a stable config: multimask_output=True + always pick masks[1]
              - Process bboxes one-by-one (avoid potential batching issues)
              - Use YOLO bboxes as SAM2 prompts to improve segmentation accuracy
              - Support multi-person mask combination
              - Auto GPU memory handling and worker count estimation
              - Resume capability via checkpoint list (completed_masks_method1_*.txt)
              - Keep post-processing like hole filling
              - Removed aspect-ratio checks; support all human poses
            
            Filtering criteria:
              - Relative width ≥ 5% (of image width)
              - Relative height ≥ 10% (of image height)
              - Relative area ≥ 1% (tunable; relative to image area)
              - Confidence ≥ 0.3 (tunable)
              - Aspect-ratio constraint: removed (accept all poses)
            
            SAM2 config:
              - multimask_output=True: generate 3 candidate masks
              - Mask selection: always choose masks[1] (empirically most stable)
              - Processing: per-bbox (avoid batching complexity)
            """
    )
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="dataset root (with train/test folders)")
    parser.add_argument("--mode", type=str, choices=["first_frame", "all_frames"],
                        default="first_frame",
                        help="mode: first_frame (first only) / all_frames (all)")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker count (manual, default = auto by GPU memory)")
    parser.add_argument("--gpu_memory", type=int, default=3500,
                        help="per-worker GPU memory (MB), default = 3500")
    parser.add_argument("--min_person_size", type=float, default=0.01,
                        help="min person relative area (default 0.01 = 1%)")
    parser.add_argument("--min_confidence", type=float, default=0.3,
                        help="min confidence for person detection (default 0.3)")
    parser.add_argument("--debug_filtering", action="store_true",
                        help="debug: detailed bbox filtering info")
    
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Err: data path not exists: {args.data_dir}")
        exit(1)

    GLOBAL_MIN_PERSON_SIZE = args.min_person_size
    GLOBAL_MIN_CONFIDENCE = args.min_confidence
    GLOBAL_DEBUG_FILTERING = args.debug_filtering
    
    # 显示配置信息
    print(f"Config:")
    print(f"Data path: {args.data_dir}")
    print(f"Mode: {args.mode}")
    print(f"Yolo path: {YOLO_MODEL_PATH}")
    print(f"SAM2 config: sam2_hiera_l.yaml")
    print(f"min_person_size: {args.min_person_size:.3f}")
    print(f"min_confidence: {args.min_confidence:.2f}")
    print(f"filter: {'Enable' if args.debug_filtering else 'Disable'}")
    if args.workers:
        print(f"Worker count: {args.workers} (manual)")
    else:
        print(f"Worker count: auto estimate (Each worker {args.gpu_memory}MB GPU memory)")

    try:
        process_dataset(
            root_dir=args.data_dir,
            first_frame_only=(args.mode == "first_frame"),
            max_workers=args.workers
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")