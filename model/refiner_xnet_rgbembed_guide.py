"""
An attemp for instance (guide)-based with identical main model structure;
Not used in this project.
"""

from typing import List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pywt
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------------------------------------------------------
# wavelet kernel (not learnable)
# -----------------------------------------------------------------------------
SUPPORTED_WAVELETS = ["db1", "db2", "db4", "db6", "db8", "sym8"]
wavelet_filters = {}
wavelet_inv_filters = {}
for name in SUPPORTED_WAVELETS:
    w = pywt.Wavelet(name)
    # dec_hi, dec_lo
    # DWT in marh：y[n] = Σ h[k] * x[n+k]
    # cnn to simulate such process：y[n] = Σ h[k] * x[n-k]
    # note that we need to transpose since we are using cnn
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    # reconstruction
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

    # LL: (horizontal lo) × (vertical lo) → LL (main information)
    # HL: (horizontal hi) × (vertical lo) → horizontal edges
    # LH: (horizontal lo) × (vertical hi) → vertical edges  
    # HH: (horizontal hi) × (vertical hi) → diagonal edges
    filt = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)[:, None]  # shape [4,1,k,k]

    # reconstruction filter
    inv = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)[:, None]  # shape [4,1,k,k]

    # make all unlearnable
    wavelet_filters[name]     = Variable(filt, requires_grad=False)
    wavelet_inv_filters[name] = Variable(inv,  requires_grad=False)


# -----------------------------------------------------------------------------
# DWT/IWT class
# -----------------------------------------------------------------------------
class WaveletDWTIWT(nn.Module):
    def __init__(self, wavelet_list: List[str]):
        super().__init__()
        self.wavelet_list = wavelet_list
        self.dwt_layers = nn.ModuleDict()
        self.iwt_layers = nn.ModuleDict()

        for name in wavelet_list:
            filt = wavelet_filters[name]      # [4,1,k,k]
            inv  = wavelet_inv_filters[name]  # [4,1,k,k]
            k = filt.size(-1)
            pad = (k - 1) // 2  # input should be able to be divisible by 2^L

            # Conv2d(in=1,out=4,stride=2)， 4 -> 4 subbands
            conv = nn.Conv2d(1, 4, k, stride=2, padding=pad, bias=False)
            conv.weight.data.copy_(filt)
            conv.weight.requires_grad_(False)
            self.dwt_layers[name] = conv

            # iwt: ConvTranspose2d(in=4,out=1,stride=2)
            deconv = nn.ConvTranspose2d(4, 1, k, stride=2, padding=pad, bias=False)
            deconv.weight.data.copy_(inv)
            deconv.weight.requires_grad_(False)
            self.iwt_layers[name] = deconv

    def multi_dwt_concat(self, x: torch.Tensor, up_mode='nearest'):
        # cascade DWT
        # no averaging all point-wise summation since information will be diluted
        # L_n input is LL from L_n-1
        # all decomposition upsample to first level resolution
        coeffs = []
        cur = x

        with torch.no_grad():
            tmp = self.dwt_layers[self.wavelet_list[0]](cur[:, :1])
            H1, W1 = tmp.shape[-2:]

        for name in self.wavelet_list:
            conv = self.dwt_layers[name]
            B, C, H, W = cur.shape
            outs = []
            for c in range(C):
                outs.append(conv(cur[:, c:c+1]))
            out = torch.cat(outs, dim=1)  # [B,4C,H/2,W/2], 4 as 4 subbands
            ll = out[:, 0::4]
            hf = torch.cat([out[:, 4*i+1:4*i+4] for i in range(C)], dim=1)
            # upsample to first level dwt outputs shape
            ll_u = F.interpolate(ll, size=(H1, W1), mode=up_mode)
            hf_u = F.interpolate(hf, size=(H1, W1), mode=up_mode)
            coeffs.append((ll_u, hf_u))
            cur = ll

        ll_cat = torch.cat([c[0] for c in coeffs], dim=1)  # [B, C*L, H//2, W//2]
        hf_cat = torch.cat([c[1] for c in coeffs], dim=1)  # [B, 3*C*L, H//2, W//2]
        return ll_cat, hf_cat

    def multi_iwt_from_concat(self,
                              ll_cat: torch.Tensor,
                              hf_cat: torch.Tensor,
                              up_mode='nearest'):
        # cascade IWT
        B, c_mul_L, H1, W1 = ll_cat.shape
        L = len(self.wavelet_list)  # cascade level amount
        C = c_mul_L // L

        ll_list = ll_cat.split(C, dim=1)
        hf_list = hf_cat.split(3*C, dim=1)

        # downsample to actual dwt resolution; since we apply upsample during dwt
        ll_rev, hf_rev = [], []
        for i, (ll_u, hf_u) in enumerate(zip(ll_list, hf_list)):
            factor = 2 ** i
            Hi, Wi = H1 // factor, W1 // factor
            ll_rev.append(F.interpolate(ll_u, size=(Hi, Wi), mode=up_mode))
            hf_rev.append(F.interpolate(hf_u, size=(Hi, Wi), mode=up_mode))

        # inverse transform
        cur = ll_rev[-1]
        for name, hf_i in zip(reversed(self.wavelet_list), reversed(hf_rev)):
            Bc, _, hi, wi = cur.shape
            # hf_i: [B,3C,hi,wi]→[B,C*3,hi,wi] then concat
            coeffs = []
            for c in range(C):
                hi_c = hf_i[:, 3*c:3*c+3]
                coeffs.append(torch.cat([cur[:, c:c+1], hi_c], dim=1))
            # inverse from n is the ll from n-1
            outs = []
            deconv = self.iwt_layers[name]
            for coeff in coeffs:
                outs.append(deconv(coeff))
            cur = torch.cat(outs, dim=1)  # (B, C, H, W) 

        return cur

def conv3x3(in_ch, out_ch, stride=1, do_pad=True):
    pad = 1 if do_pad else 0
    return nn.Conv2d(
        in_ch, out_ch, 3, stride=stride,
        padding=pad, padding_mode="reflect",
        bias=False
    )


def down_conv(in_channels, out_channels):
    """
    3x3 conv w/ stride=2 for spatial downsampling + BN + ReLU
    """
    return nn.Sequential(
        nn.ReflectionPad2d(1),    
        conv3x3(in_channels, out_channels, stride=2, do_pad=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def same_conv(in_channels, out_channels):
    """
    3x3 conv w/ stride=1 (same resolution) + BN + ReLU
    """
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_shuffle(in_ch, r=2):
    """PixelShuffle upsampling: Conv→PixelShuffle→BN→ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch * r * r, 3, 1, 1, bias=False),
        nn.PixelShuffle(r),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    """
    ResNet-style basic block: two 3x3 convs + BN + ReLU with residual connection;
    x --> conv1 --> BN --> ReLU --> conv2 --> BN --> (+) --> ReLU
    |                                                  ^
    |--------> downsample (if needed) -----------------|
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # downsample to apply 1x1 conv or other transformation to the identity branch
        # to avoid input and output dimension mismatch
        self.downsample = downsample

    def forward(self, x):
        identity = x  # shape: (B, in_channels, H, W)

        out = self.relu(self.bn1(self.conv1(x))) # (B, out_channels, H/stride, W/stride)
        out = self.bn2(self.conv2(out))  # (B, out_channels, H/stride, W/stride)

        if self.downsample is not None:
            identity = self.downsample(x)  # (B, out_channels, H/stride, W/stride)

        out += identity  # (B, out_channels, H/stride, W/stride)
        out = self.relu(out)
        return out


class HFSE(nn.Module):
    def __init__(self, C_in, r=8):
        """
        C_in = 3*C*L      (must be divisible by 3; since HF has 3 subbands)
        r    = reduction
        """
        super().__init__()
        self.Cg = C_in // 3           # group number = C*L
        r_eff = 2 ** int(math.log2(self.Cg)) if self.Cg >= 1 else 1
        reduced = self.Cg // r_eff
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv1d(self.Cg, reduced, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(reduced, self.Cg, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape          # C = 3 * Cg
        x = x.view(B, self.Cg, 3, H, W)

        w = self.avg(x)                              # (B, Cg, 3, 1, 1)
        w = self.fc(w.squeeze(-1).squeeze(-1))       # (B, Cg, 3)

        w = w.mean(-1, keepdim=True)                 # (B, Cg, 1)

        w = w.view(B, self.Cg, 1, 1, 1)              # (B, Cg, 1, 1, 1)
        out = x * w                                  # broadcast 乘权
        return out.view(B, C, H, W)


class RGDF(nn.Module):
    def __init__(self, channels, use_edge=False):
        super().__init__()
        self.use_edge = use_edge
        self.channels = channels

        # channel-wise attention: independent modeling of each directional component
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # residual fusion: concatenate along directional dimension then 
        # conv compression and fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        if use_edge:
            self.edge_proj = nn.Conv2d(1, channels, 1, bias=False)

    def forward(self, hf_cat, edge_map=None):
        B, C3, H, W = hf_cat.shape
        C = C3 // 3

        hf_split = hf_cat.view(B, C, 3, H, W).permute(2, 0, 1, 3, 4)  # → [3, B, C, H, W]

        ca_weighted = []
        for i in range(3):
            feat = hf_split[i]  # [B, C, H, W]
            w = self.ca(feat)
            if self.use_edge and edge_map is not None:
                edge_weight = self.edge_proj(edge_map)  # [B, C, H, W]
                w = w * edge_weight.sigmoid()
            ca_weighted.append(feat * w)

        fused = torch.cat(ca_weighted, dim=1)  # [B, 3C, H, W]
        out = self.fuse(fused) + hf_cat[:, :C]
        return out.repeat(1, 3, 1, 1)


class BranchEncoder(nn.Module):
    """
    Encoder (single branch);
    In final model, two branches are identity, just processing different information
    """
    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 num_layers: int,
                 blocks_per_layer: int = 2):
        super(BranchEncoder, self).__init__()
        self.num_layers = num_layers
        channels = [base_channels * (2 ** i) for i in range(num_layers)]
        self.init_conv = same_conv(in_channels, channels[0])
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()

        prev_channels = channels[0]
        for idx, out_channels in enumerate(channels):
            if idx > 0:
                self.downs.append(down_conv(prev_channels, out_channels))
                prev_channels = out_channels
            blocks = []
            for b in range(blocks_per_layer):
                downsample = None
                if b == 0 and idx > 0 and prev_channels != out_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(prev_channels, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                blocks.append(BasicBlock(prev_channels, out_channels, downsample=downsample))
            self.stages.append(nn.Sequential(*blocks))
        self.channels = channels

    def forward(self, x):
        feats = []
        out = self.init_conv(x)
        for idx in range(self.num_layers):
            if idx > 0:
                out = self.downs[idx-1](out)
            out = self.stages[idx](out)
            feats.append(out)
        return feats


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)  # (B, in_channels, H, W)
        out = self.conv2(self.conv1(x))  # (B, out_channels, H, W)
        out += identity
        return self.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = up_shuffle(in_channels, r=2)
        self.conv = nn.Sequential(
            ResidualBlock(in_channels + skip_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        # (B, in_channels, H, W) -> (B, in_channels, H*2, W*2)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # (B, in_channels+skip_channels, H*2, W*2)
        return self.conv(x)  # (B, out_channels, H, W)


class BranchDecoder(nn.Module):
    def __init__(self, encoder_channels, base_channels, out_channels, enhance_structure=None):
        super().__init__()

        # (option) all specified layers (including the deepest layer) 
        # underwent channel doubling
        # not used in the final model
        # encoder_channels: 
        # [base_channels*2^0, base_channels*2^1, ..., base_channels*2^(num_layers-1)]
        decoder_channels = encoder_channels.copy()
        if enhance_structure:
            for i, factor in enhance_structure.items():
                decoder_channels[i] *= factor

        in_ch = decoder_channels[-1]  # on the deepest layer
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 2, -1, -1):
            skip_ch = decoder_channels[i]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, skip_ch))
            in_ch = skip_ch
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, feats):
        dec_feats = []
        x = feats[-1]  # (B, base_channels * 2^(num_layers-1), H_L, W_L)
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, feats[-(i + 2)])  # B, decoder_channels[-(i+2)], H_{L-i-1}, W_{L-i-1}
            dec_feats.append(x)
        return dec_feats, self.final_conv(x)


def build_hf_enhance_structure(num_layers, layers_to_enhance=(2, 3), factor=2):
    # for channel doubling to specify the layers that needed to be enhanced
    # not used in the dinal model
    return {num_layers - 1 - l: factor for l in layers_to_enhance}

class CrossAxisAttention(nn.Module):
    """
    Cross-Axis Attention  
    Supports 4 directions：
        ─ row     : same row slice
        ─ col     : same column slice
        ─ diag    : same main diagonal slice
        ─ global  : whole image
    Using main diagonal slice rather than all diagonal ones is to incur one reduction in throughput
    """
    def __init__(self, C_l: int, C_h: int, d_head: int = 128,
                 modes: set = {'row', 'col', 'diag', 'global'}):
        super().__init__()
        self.modes = modes

        # ----------- Q K V projection ----------- #
        self.qh = nn.Conv2d(C_h, d_head, 1, bias=False)
        self.kh = nn.Conv2d(C_h, d_head, 1, bias=False)
        self.vh = nn.Conv2d(C_h, d_head, 1, bias=False)

        self.ql = nn.Conv2d(C_l, d_head, 1, bias=False)
        self.kl = nn.Conv2d(C_l, d_head, 1, bias=False)
        self.vl = nn.Conv2d(C_l, d_head, 1, bias=False)

        # ----------- out projection ----------- #
        self.proj_h = nn.Conv2d(d_head, C_h, 1)
        self.proj_l = nn.Conv2d(d_head, C_l, 1)
        self.aux_h  = nn.Conv2d(C_h,   C_h, 1)
        self.aux_l  = nn.Conv2d(C_l,   C_l, 1)

        # learnable weight for all directions
        self.alpha = nn.Parameter(torch.ones(4))
        self.direction2idx = {'row': 0, 'col': 1, 'diag': 2, 'global': 3}

        # index buffer to accelerate diagonal fetech：
        # K=(H,W), V=(diag_idx, pos_idx)
        self._diag_cache: dict[tuple[int,int],
                               tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # generate index buffer for diagonal
    # ------------------------------------------------------------------ #
    def _diag_idx(self, H: int, W: int, device: torch.device):
        key = (H, W)
        if key not in self._diag_cache:
            i = torch.arange(H).view(H, 1).expand(H, W)         # [H,W]
            j = torch.arange(W).view(1, W).expand(H, W)         # [H,W]
            diag = (j - i + (H - 1)).reshape(-1).long()         # indexing along the main diagonal
            pos  = torch.where(j >= i, i, j).reshape(-1).long() # partial ordering on this diagonal
            self._diag_cache[key] = (diag, pos)                 # cached in the CPU

        diag_cpu, pos_cpu = self._diag_cache[key]
        # loaded onto the current GPU every time it is called
        # to avoid conflicts under multi-GPU settings
        return diag_cpu.to(device, non_blocking=True), pos_cpu.to(device, non_blocking=True)

    def forward(self, ll, hf, edge=None):
        B, C_l, H, W = ll.shape
        _, C_h, _, _ = hf.shape
        d   = self.qh.out_channels
        N   = H * W
        scl = d ** -0.5  # scaled dot-product 

        # Q K V on global feature map
        if edge is None:
            Qh = self.qh(hf).view(B, d, N)
        else:
            # edge : [B,1,H,W] 
            edge.detach()
            Qh = self.qh(edge.expand(-1, hf.size(1), -1, -1)).contiguous().view(B, d, N)
        Kh = self.kh(hf).view(B, d, N)
        Vh = self.vh(hf).view(B, d, N)
        Ql = self.ql(ll).view(B, d, N)
        Kl = self.kl(ll).view(B, d, N)
        Vl = self.vl(ll).view(B, d, N)

        # placeholder
        z_ll = lambda: torch.zeros_like(ll)
        z_hf = lambda: torch.zeros_like(hf)

        # norm directional weights
        wdir = torch.softmax(self.alpha, dim=0)

        out_ll_list, out_hf_list = [], []

        # ---------------- Global Attention ---------------- #
        if 'global' in self.modes:
            OL = F.scaled_dot_product_attention(
                Qh.transpose(1, 2), Kl.transpose(1, 2), Vl.transpose(1, 2)
            ).transpose(1, 2)                            # (B,d,N)
            OH = F.scaled_dot_product_attention(
                Ql.transpose(1, 2), Kh.transpose(1, 2), Vh.transpose(1, 2)
            ).transpose(1, 2)
            out_ll_list.append(self.proj_l(OL.view(B, d, H, W)) *
                               wdir[self.direction2idx['global']])
            out_hf_list.append(self.proj_h(OH.view(B, d, H, W)) *
                               wdir[self.direction2idx['global']])
        else:
            out_ll_list.append(z_ll()); out_hf_list.append(z_hf())

        # ---------------- Row Attention ------------------- #
        if 'row' in self.modes:
            Qh_r = Qh.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Kl_r = Kl.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Vl_r = Vl.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Ar   = torch.softmax(Qh_r @ Kl_r.transpose(-2, -1) * scl, dim=-1)
            Lr   = (Ar @ Vl_r).reshape(B, H, W, d).permute(0, 3, 1, 2)
            out_ll_list.append(self.proj_l(Lr) * wdir[self.direction2idx['row']])

            Ql_r = Ql.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Kh_r = Kh.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Vh_r = Vh.view(B, d, H, W).permute(0, 2, 3, 1).reshape(B*H, W, d)
            Ah   = torch.softmax(Ql_r @ Kh_r.transpose(-2, -1) * scl, dim=-1)
            Hr   = (Ah @ Vh_r).reshape(B, H, W, d).permute(0, 3, 1, 2)
            out_hf_list.append(self.proj_h(Hr) * wdir[self.direction2idx['row']])
        else:
            out_ll_list.append(z_ll()); out_hf_list.append(z_hf())

        # ---------------- Column Attention ---------------- #
        if 'col' in self.modes:
            Qh_c = Qh.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Kl_c = Kl.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Vl_c = Vl.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Ac   = torch.softmax(Qh_c @ Kl_c.transpose(-2, -1) * scl, dim=-1)
            Lc   = (Ac @ Vl_c).reshape(B, W, H, d).permute(0, 3, 2, 1)
            out_ll_list.append(self.proj_l(Lc) * wdir[self.direction2idx['col']])

            Ql_c = Ql.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Kh_c = Kh.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Vh_c = Vh.view(B, d, H, W).permute(0, 3, 2, 1).reshape(B*W, H, d)
            Ahc  = torch.softmax(Ql_c @ Kh_c.transpose(-2, -1) * scl, dim=-1)
            Hc   = (Ahc @ Vh_c).reshape(B, W, H, d).permute(0, 3, 2, 1)
            out_hf_list.append(self.proj_h(Hc) * wdir[self.direction2idx['col']])
        else:
            out_ll_list.append(z_ll()); out_hf_list.append(z_hf())

        # ---------------- Diagonal Attention -------------- #
        if 'diag' in self.modes:
            flat_idx, _ = self._diag_idx(H, W, hf.device)      # (N,)

            # helper: SDPA → (B,d,N)
            def attn(q, k, v):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)

            Ld = attn(Qh[:, :, flat_idx], Kl[:, :, flat_idx], Vl[:, :, flat_idx])  # (B,d,N)
            Hd = attn(Ql[:, :, flat_idx], Kh[:, :, flat_idx], Vh[:, :, flat_idx])  # (B,d,N)

            # scatter back to (B,d,N)
            blank_l = torch.zeros(B, d, N, device=ll.device, dtype=Ld.dtype)
            blank_l.scatter_(2,
                             flat_idx.unsqueeze(0).unsqueeze(0).expand(B, d, -1),
                             Ld)
            blank_h = torch.zeros(B, d, N, device=hf.device, dtype=Hd.dtype)
            blank_h.scatter_(2,
                             flat_idx.unsqueeze(0).unsqueeze(0).expand(B, d, -1),
                             Hd)

            out_ll_list.append(self.proj_l(blank_l.view(B, d, H, W)) *
                               wdir[self.direction2idx['diag']])
            out_hf_list.append(self.proj_h(blank_h.view(B, d, H, W)) *
                               wdir[self.direction2idx['diag']])
        else:
            out_ll_list.append(z_ll()); out_hf_list.append(z_hf())

        # ---------------- residual and helper conv --------------- #
        ll_out = self.aux_l(ll + sum(out_ll_list))
        hf_out = self.aux_h(hf + sum(out_hf_list))
        return ll_out, hf_out


class EnhancedPostHeadResidual(nn.Module):
    def __init__(self, feat_ch, rgb_ch=3, mid_ch=128, use_rgb=True):
        super().__init__()
        self.use_rgb = use_rgb

        # projecting recon_raw onto the intermediate channel space
        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # local branch: responsible for capturing local structural features
        # with group convolution preserving spatial textures
        self.local_branch = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch // 4, bias=False)

        # global branch：applying channel attention (akin to squeeze-excitation)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.resblock = ResidualBlock(mid_ch, mid_ch)

        # fuse and output delta
        in_fuse_ch = mid_ch + rgb_ch if use_rgb else mid_ch
        self.fuse = nn.Conv2d(in_fuse_ch, feat_ch, kernel_size=1, bias=True)

        self.act_out = nn.Hardsigmoid()

    def forward(self, recon_raw, rgb=None):
        x = self.proj(recon_raw)  # B, mid_ch, H, W

        local_feat = self.local_branch(x)                   
        global_weight = self.global_branch(x)               
        fused_feat = local_feat * global_weight + local_feat  

        res_feat = self.resblock(fused_feat)

        # (option) RGB guided
        # not used
        if self.use_rgb and rgb is not None:
            res_feat = torch.cat([res_feat, rgb], dim=1)

        delta = self.fuse(res_feat)
        return self.act_out(recon_raw + delta)


class ShallowDilatedStem(nn.Module):
    """
    maintain the H×W dilated convolution pathway for retaining 
    fine details at the 1–2 pixel level
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1, dilation=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(True),
            nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(True),
            nn.Conv2d(32, out_ch, 3, 1, 3, dilation=3, bias=False),  # → out_ch = base_channels
        )
    def forward(self, x):
        return self.convs(x)

class ResidualGuideGate(nn.Module):
    """
    y = x + gain * ( Delta(x) * Mhat )
    - gain ∈ [0, max_gain], ensures "weak guidance"
    - Mhat smooths the guide and is detached to prevent over-dependence / gradient leakage
    - depthwise allows each channel to be fine-tuned independently (natural for 3*L HF sub-bands)
    """
    def __init__(self, channels, guide_ch=1, max_gain=0.15, grouped=True, 
                 blur_k=5, temp=2.0, drop_prob=0.0):
        super().__init__()
        groups = channels if grouped else 1
        self.delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, groups=groups, bias=False),
        )
        self.to_gate = nn.Conv2d(guide_ch, 1, 1, bias=True)  # soft gate generation (1 channel)
        self.temp = temp
        self.blur = nn.AvgPool2d(blur_k, stride=1, padding=blur_k//2)
        self.max_gain = max_gain
        # initialize with a small value to avoid lazy dependence on the guide 
        # before the model starts learning
        self.gain_logit = nn.Parameter(torch.tensor(-4.0))
        self.drop_prob = drop_prob

    def forward(self, x, guide):
        m = torch.sigmoid(self.to_gate(guide) / self.temp)
        m = self.blur(m).detach()

        # optional mask dropout during training (to prevent overfitting to the guide)
        if self.training and self.drop_prob > 0:
            if torch.rand(1, device=x.device).item() < self.drop_prob:
                m = torch.zeros_like(m)

        delta = self.delta(x) * m   
        gain  = self.max_gain * torch.sigmoid(self.gain_logit)
        return x + gain * delta

# -----------------------------------------------------------------------------
# main architecture
# -----------------------------------------------------------------------------
class refiner_xnet(nn.Module):
    def __init__(self,
                 wavelet_list=("db1","db2","db4"),
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_layers: int = 4,
                 blocks_per_layer: int = 2,
                 d_head: int = 128, 
                 modes_bn={'row','col','diag','global'},
                 modes_mid={'row','col','global'}
                ):
        super().__init__()
        self.wave = WaveletDWTIWT(wavelet_list)
        L = len(wavelet_list)

        ll_in_ch = in_channels + in_channels * L          # = 3 + 3L
        hf_in_ch = in_channels + 3 * in_channels * L      # = 3 + 9L

        # spatial-wise
        self.sa_hf = nn.Sequential(
            nn.Conv2d(3*in_channels*L, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # channel-wise
        self.prefuse_hf = RGDF(channels=3*in_channels*L // 3, use_edge=True)

        self.stem_shallow = ShallowDilatedStem(
            hf_in_ch, base_channels,
        )

        # ── dual-branch encoder ──
        self.encoder_ll = BranchEncoder(ll_in_ch, base_channels,
                                        num_layers, blocks_per_layer)
        self.encoder_hf = BranchEncoder(hf_in_ch, base_channels,
                                        num_layers, blocks_per_layer)

        encoder_channels = [base_channels * (2 ** i)      # [C,2C,4C,8C]
                            for i in range(num_layers)]

        self.cross_attn_bn = CrossAxisAttention(
            C_l=encoder_channels[-1],     # deepest layer
            C_h=encoder_channels[-1],
            d_head=d_head,
            modes=modes_bn,
            )
        self.cross_attn_mid = CrossAxisAttention(
            C_l=encoder_channels[-2],     # the second-to-last layer
            C_h=encoder_channels[-2],
            d_head=d_head//2,
            modes=modes_mid,
            )

        # ── dual branch decoder ──
        self.decoder_ll = BranchDecoder(encoder_channels, base_channels, 1 * L)
        self.decoder_hf = BranchDecoder(encoder_channels, base_channels,
                                        3 * 1 * L, enhance_structure=None)

        # soft gates on ll and hf
        self.ll_gate = ResidualGuideGate(
            channels=1*L,  max_gain=0.15, grouped=True, drop_prob=0.1
        )
        self.hf_gate = ResidualGuideGate(
            channels=3*1*L, max_gain=0.15, grouped=True, drop_prob=0.1
        )
        self.shallow2ll = nn.Conv2d(base_channels, 1*L, 1, bias=False)
        self.shallow2hf = nn.Conv2d(base_channels, 3*1*L, 1, bias=False)

        # BN‐level cross‐attn
        self.attn_bn_head_ll  = nn.Conv2d(encoder_channels[-1], 1, kernel_size=1, bias=False)
        self.attn_bn_head_hf  = nn.Conv2d(encoder_channels[-1], 1, kernel_size=1, bias=False)
        # Mid‐level cross‐attn 
        self.attn_mid_head_ll = nn.Conv2d(encoder_channels[-2], 1, kernel_size=1, bias=False)
        self.attn_mid_head_hf = nn.Conv2d(encoder_channels[-2], 1, kernel_size=1, bias=False)
        # shallow branch (not used)
        self.shallow_head     = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)

        # post head
        self.post = EnhancedPostHeadResidual(
            feat_ch=1, rgb_ch=3, mid_ch=64, use_rgb=False
        )


    # -----------------------------------------------------------------
    def forward(self, rgb: torch.Tensor, guide_mask: torch.Tensor):
        # slightly sharpen rgb input
        alpha = 0.7
        blur  = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2)
        rgb_sharp = rgb + alpha * (rgb - blur)

        # ---------- DWT ----------
        ll_cat, hf_cat = self.wave.multi_dwt_concat(rgb_sharp)
        edge_map = hf_cat.abs().mean(1, keepdim=True)

        hf_cat = self.prefuse_hf(hf_cat, edge_map)  
        hf_cat = hf_cat * (1.0 + self.sa_hf(hf_cat))

        H1, W1 = ll_cat.shape[-2:]
        rgb_ds = F.interpolate(rgb_sharp, size=(H1, W1),
                               mode='bilinear', align_corners=False)

        in_ll = torch.cat([rgb_ds, ll_cat], dim=1)
        in_hf = torch.cat([rgb_ds, hf_cat], dim=1)

        feats_ll = self.encoder_ll(in_ll)   # list len = num_layers
        feats_hf = self.encoder_hf(in_hf)

        # ——— BN‐level cross‐attn ———
        edge_bn = F.interpolate(edge_map, size=feats_ll[-1].shape[-2:], mode='nearest')
        attn_bn_ll, attn_bn_hf = self.cross_attn_bn(feats_ll[-1], feats_hf[-1], edge_bn)
        feats_ll[-1], feats_hf[-1] = attn_bn_ll, attn_bn_hf
        pred_attn_bn_ll = self.attn_bn_head_ll(attn_bn_ll)
        pred_attn_bn_hf = self.attn_bn_head_hf(attn_bn_hf)
        # ——— mid‐level cross‐attn ———
        edge_mid = F.interpolate(edge_map, size=feats_ll[-2].shape[-2:], mode='nearest')
        attn_mid_ll, attn_mid_hf = self.cross_attn_mid(feats_ll[-2], feats_hf[-2], edge_mid)
        feats_ll[-2], feats_hf[-2] = attn_mid_ll, attn_mid_hf
        pred_attn_mid_ll = self.attn_mid_head_ll(attn_mid_ll)
        pred_attn_mid_hf = self.attn_mid_head_hf(attn_mid_hf)

        # ---------- decode & IWT ----------
        dec_feats_ll, dec_ll = self.decoder_ll(feats_ll)      
        dec_feats_hf, dec_hf = self.decoder_hf(feats_hf)

        # ---------- shallow branch ----------
        # not used
        hf_cat_up   = F.interpolate(hf_cat, size=rgb_sharp.shape[-2:], mode='nearest')
        shallow_feat = self.stem_shallow(torch.cat([rgb_sharp, hf_cat_up], 1))
        shallow_ds = F.interpolate(
            shallow_feat, size=dec_ll.shape[-2:],
            mode='bilinear', align_corners=False)
        pred_shallow   = self.shallow_head(shallow_ds)

        guide_ds = F.interpolate(
            guide_mask, size=dec_ll.shape[-2:], mode='bilinear', align_corners=False
        )
        dec_ll = self.ll_gate(dec_ll, guide_ds)
        dec_hf = self.hf_gate(dec_hf, guide_ds)

        recon_raw = self.wave.multi_iwt_from_concat(dec_ll, dec_hf)
        recon     = self.post(recon_raw, None)

        return (feats_ll, feats_hf, ll_cat, hf_cat,
                dec_feats_ll, dec_feats_hf, pred_shallow, 
                pred_attn_bn_ll, pred_attn_bn_hf, pred_attn_mid_ll, 
                pred_attn_mid_hf, recon_raw, recon)



if __name__ == "__main__":
    # test model
    model = refiner_xnet()
    # pseudo input：batch=1, 3*720*1280
    dummy_input = torch.randn(2, 3, 720, 1280)
    dummy_guide = torch.randn(2, 1, 720, 1280)
    feats_ll, feats_hf, ll_cat, hf_cat, dec_feats_ll, dec_feats_hf, \
    shallow_ds, attn_bn_ll, attn_bn_hf, attn_mid_ll, \
    attn_mid_hf, recon_raw, recon = model(dummy_input, dummy_guide)
    print("Low-frequency branch feature maps:")
    for i, f in enumerate(feats_ll):
        print(f"  Layer {i}: {tuple(f.shape)}")
    print("High-frequency branch feature maps:")
    for i, f in enumerate(feats_hf):
        print(f"  Layer {i}: {tuple(f.shape)}")

    print("Low-frequency decoder branch feature maps:")
    for i, f in enumerate(dec_feats_ll):
        print(f"  Layer {i}: {tuple(f.shape)}")
    print("High-frequency decoder branch feature maps:")
    for i, f in enumerate(dec_feats_hf):
        print(f"  Layer {i}: {tuple(f.shape)}")

    print(f"IWT image shape: {tuple(recon_raw.shape)}")
    print(f"Reconstructed image shape: {tuple(recon.shape)}")