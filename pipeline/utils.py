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