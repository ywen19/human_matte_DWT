import csv
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.datapipes.iter import IterableWrapper, ShardingFilter, Mapper
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def split_csv(csv_path, split_ratio=0.8, seed=42):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    random.seed(seed)
    random.shuffle(samples)

    split_point = int(len(samples) * split_ratio)
    return samples[:split_point], samples[split_point:]


def make_trimap_distance(
    alpha_np,
    fg_thresh=0.95,
    bg_thresh=0.05,
    unknown_radius=20
):
    """
    Distance Transform for trimap generation;
    
    Args:
        alpha_np:       gt alpha ranged within 0-1
        fg_thresh:      thresholder; values greater than this will be FG
        bg_thresh:      thresholder; values lesser than this will be BG
        unknown_radius: control the band width of trimap

    Returns:
        trimap: 0-BG, 0.5-unknown, 1.0-FG
    """
    h, w = alpha_np.shape

    # binarize alpha gt
    fg_bin = (alpha_np > fg_thresh).astype(np.uint8)   
    bg_bin = (alpha_np < bg_thresh).astype(np.uint8)  

    # distanceTransform 
    dist_to_fg = cv2.distanceTransform(1 - fg_bin, cv2.DIST_L2, 5)
    dist_to_bg = cv2.distanceTransform(1 - bg_bin, cv2.DIST_L2, 5)

    trimap = np.full((h, w), 128, dtype=np.uint8) # all 128(0.5) for trimap init

    # if BG distance>unknown_radius, then the pixel belongs to FG;
    # if FG distance>unknown_radius, then the pixel belongs to BG;
    trimap[dist_to_bg > unknown_radius] = 255
    trimap[dist_to_fg > unknown_radius] = 0

    return trimap


def resize_tensor(tensor, size, mode):
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode=mode, align_corners=False
    ).squeeze(0)


def build_iterable_datapipe(
    sample_list,
    resize_to=(720, 1280),
    fg_thresh=0.98,
    bg_thresh=0.02,
    unknown_radius=40,
    shuffle=True,
    seed=42,
    do_crop=False,
    crop_size=(512, 512)
):
    if shuffle:
        rng = np.random.default_rng(seed)
        sample_list = rng.permutation(sample_list).tolist()

    pipe = IterableWrapper(sample_list)
    pipe = ShardingFilter(pipe)

    def load_resize_tensorize(sample):
        rgb = to_tensor(Image.open(sample['rgb']).convert('RGB'))      # [3, H, W]
        mask = to_tensor(Image.open(sample['init_mask']).convert('L')) # [1, H, W]
        gt   = to_tensor(Image.open(sample['gt']).convert('L'))        # [1, H, W]

        # resize
        th, tw = resize_to
        rgb = resize_tensor(rgb, (th, tw), mode='bilinear')
        mask = resize_tensor(mask, (th, tw), mode='bilinear')
        gt   = resize_tensor(gt,   (th, tw), mode='bilinear')

        # (option) random crop
        if do_crop:
            ch, cw = crop_size
            _, H, W = rgb.shape
            top  = random.randint(0, max(0, H - ch))
            left = random.randint(0, max(0, W - cw))
            rgb  = rgb[:,  top:top+ch,   left:left+cw]
            mask = mask[:, top:top+ch,   left:left+cw]
            gt   = gt[:,   top:top+ch,   left:left+cw]
            
        rgb = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
        # trimap generation
        gt_np = gt.squeeze(0).cpu().numpy().astype(np.float32) 
        trimap_np = make_trimap_distance(
            alpha_np=gt_np,
            fg_thresh=fg_thresh,
            bg_thresh=bg_thresh,
            unknown_radius=unknown_radius
        )
        # from (0, 255) range back to (0, 1)
        trimap = torch.from_numpy(trimap_np).unsqueeze(0).float() / 255.0

        return rgb, mask, gt, trimap

    return Mapper(pipe, load_resize_tensorize)


