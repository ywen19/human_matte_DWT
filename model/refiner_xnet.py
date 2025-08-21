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
    # 分解滤波器（高低频），注意反转 dec_hi, dec_lo
    # 小波变换在数学上定义为相关操作：y[n] = Σ h[k] * x[n+k]
    # Conv2d执行卷积操作：y[n] = Σ h[k] * x[n-k]
    # 卷积 = 相关 + 滤波器反转，所以要预先反转滤波器来补偿这个差异
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    # 重构滤波器, 不需要反转，因为重构过程的数学定义与Conv2d一致
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

    # 分解时 2D 小波共轭对乘，4 个子带: LL, HL, LH, HH
    # LL: 水平低频 × 垂直低频 → 近似分量（包含主要信息）
    # HL: 水平高频 × 垂直低频 → 水平边缘
    # LH: 水平低频 × 垂直高频 → 垂直边缘  
    # HH: 水平高频 × 垂直高频 → 对角边缘
    filt = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)[:, None]  # shape [4,1,k,k]

    # 2D逆小波变换的4个重构滤波器
    inv = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)[:, None]  # shape [4,1,k,k]

    # 将滤波器转换为不可训练的Variable（固定权重）
    wavelet_filters[name]     = Variable(filt, requires_grad=False)
    wavelet_inv_filters[name] = Variable(inv,  requires_grad=False)


# -----------------------------------------------------------------------------
# 2. 封装成 Conv2d/ConvTranspose2d的DWT/IWT
# 封装成pytorch模块也是为了更高效的GPU并行
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
            pad = (k - 1) // 2  # 避免输入不能被2^L整除

            # 分解：Conv2d(in=1,out=4,stride=2)， 4是指4个子频带支路
            conv = nn.Conv2d(1, 4, k, stride=2, padding=pad, bias=False)
            conv.weight.data.copy_(filt)
            conv.weight.requires_grad_(False)
            self.dwt_layers[name] = conv

            # 重构：ConvTranspose2d(in=4,out=1,stride=2)
            deconv = nn.ConvTranspose2d(4, 1, k, stride=2, padding=pad, bias=False)
            deconv.weight.data.copy_(inv)
            deconv.weight.requires_grad_(False)
            self.iwt_layers[name] = deconv

    def multi_dwt_concat(self, x: torch.Tensor, up_mode='nearest'):
        # 级联拼接
        # 不选平均或按位相加是因为会压缩信息
        # L_n层的输入是L_n-1层分解的ll
        # 所有的分解输出都上采样到与第一层的分辨率相同[H/2, W/2]
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
        """
        从 ll_cat,hf_cat 恢复到原始分辨率
        """
        B, c_mul_L, H1, W1 = ll_cat.shape
        L = len(self.wavelet_list)  # 级联级数
        C = c_mul_L // L

        # 拆分
        ll_list = ll_cat.split(C, dim=1)
        hf_list = hf_cat.split(3*C, dim=1)

        # 回到每级真实尺寸(下采样)
        ll_rev, hf_rev = [], []
        for i, (ll_u, hf_u) in enumerate(zip(ll_list, hf_list)):
            factor = 2 ** i
            Hi, Wi = H1 // factor, W1 // factor
            ll_rev.append(F.interpolate(ll_u, size=(Hi, Wi), mode=up_mode))
            hf_rev.append(F.interpolate(hf_u, size=(Hi, Wi), mode=up_mode))

        # 逆变换
        cur = ll_rev[-1]
        for name, hf_i in zip(reversed(self.wavelet_list), reversed(hf_rev)):
            # 拼接 1 低频 + 3 高频
            Bc, _, hi, wi = cur.shape
            # 将 hf_i 从 [B,3C,hi,wi]→[B,C*3,hi,wi] 并 concat
            coeffs = []
            for c in range(C):
                hi_c = hf_i[:, 3*c:3*c+3]
                coeffs.append(torch.cat([cur[:, c:c+1], hi_c], dim=1))
            # 对每通道块分别 deconv 再 concat
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
    """PixelShuffle 上采样: Conv→PixelShuffle→BN→ReLU"""
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
        # downsample 用于在输入和输出维度不匹配时，对 identity 分支做 1x1 卷积或其他变换
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

class LFSE(nn.Module):
    def __init__(self, C_in, r=8):
        """
        C_in = 输入通道数（例如 3*L）
        r    = reduction ratio
        """
        super().__init__()
        r_eff = max(1, 2 ** int(math.log2(C_in)))
        reduced = max(1, C_in // r_eff)

        self.avg = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Conv1d(C_in, reduced, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(reduced, C_in, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape  # C = C_in = 3 × L
        w = self.avg(x).view(B, C, 1)   # [B, C, 1, 1] → [B, C, 1]
        w = self.fc(w).view(B, C, 1, 1) # [B, C, 1]
        return x * w  

class HFSE(nn.Module):
    def __init__(self, C_in, r=8):
        """
        C_in = 3*C*L      ( 必须是 3 的倍数 )
        r    = reduction  ( 如 8 / 16 )
        """
        super().__init__()
        self.Cg = C_in // 3           # group 数 = C*L
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

        # ------ NEW: 把 3 个子带的权重再做平均，得到 1 个标量 ------
        w = w.mean(-1, keepdim=True)                 # (B, Cg, 1)

        w = w.view(B, self.Cg, 1, 1, 1)              # (B, Cg, 1, 1, 1)
        out = x * w                                  # broadcast 乘权
        return out.view(B, C, H, W)

class RGDF(nn.Module):
    def __init__(self, channels, use_edge=False):
        super().__init__()
        self.use_edge = use_edge
        self.channels = channels

        # 通道注意力：每个方向分量独立建模
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 残差融合：方向维度拼接后 → 卷积压缩融合
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

        # 每个方向子带做 CA（共享通道结构）
        ca_weighted = []
        for i in range(3):
            feat = hf_split[i]  # [B, C, H, W]
            w = self.ca(feat)
            if self.use_edge and edge_map is not None:
                edge_weight = self.edge_proj(edge_map)  # [B, C, H, W]
                w = w * edge_weight.sigmoid()
            ca_weighted.append(feat * w)

        fused = torch.cat(ca_weighted, dim=1)  # [B, 3C, H, W]
        out = self.fuse(fused) + hf_cat[:, :C]  # 残差连接（默认与第一个方向保持一致）
        return out.repeat(1, 3, 1, 1)

class BranchEncoder(nn.Module):
    """
    动态多层编码器：以 base_channels 为基础通道数，后续各层通道数按深度自动计算，
    层数由 num_layers 决定，每层包含相同数量的残差块。
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

        # encoder_channels: [base_channels*2^0, base_channels*2^1, ..., base_channels*2^(num_layers-1)]
        decoder_channels = encoder_channels.copy()
        if enhance_structure:
            for i, factor in enhance_structure.items():
                decoder_channels[i] *= factor  # 对所有指定层（包括最深层）进行了通道倍增

        in_ch = decoder_channels[-1]  # 最深层 input_channels 也被增强为 C*factor
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
        # return self.final_conv(x)

def build_hf_enhance_structure(num_layers, layers_to_enhance=(2, 3), factor=2):
    return {num_layers - 1 - l: factor for l in layers_to_enhance}

class CrossAxisAttention(nn.Module):
    """
    Cross-Axis Attention  
    支持四种方向：
        ─ row     : 同一行 (H 个序列, 长 W)
        ─ col     : 同一列 (W 个序列, 长 H)
        ─ diag    : 主/副对角线 (H+W-1 条序列)
        ─ global  : 整幅图展开 (H*W)
    `modes` 传入集合，例如 {'row','diag','global'} 表示开启这三条通路。
    若想先排除对角线，只要把 'diag' 从集合里去掉即可。
    """
    def __init__(self, C_l: int, C_h: int, d_head: int = 128,
                 modes: set = {'row', 'col', 'diag', 'global'}):
        super().__init__()
        self.modes = modes

        # ----------- Q K V 投影 ----------- #
        self.qh = nn.Conv2d(C_h, d_head, 1, bias=False)
        self.kh = nn.Conv2d(C_h, d_head, 1, bias=False)
        self.vh = nn.Conv2d(C_h, d_head, 1, bias=False)

        self.ql = nn.Conv2d(C_l, d_head, 1, bias=False)
        self.kl = nn.Conv2d(C_l, d_head, 1, bias=False)
        self.vl = nn.Conv2d(C_l, d_head, 1, bias=False)

        # ----------- 输出投影 ----------- #
        self.proj_h = nn.Conv2d(d_head, C_h, 1)
        self.proj_l = nn.Conv2d(d_head, C_l, 1)
        self.aux_h  = nn.Conv2d(C_h,   C_h, 1)
        self.aux_l  = nn.Conv2d(C_l,   C_l, 1)

        # 可学习方向权重 (row/col/diag/global 对应 0-3)
        self.alpha = nn.Parameter(torch.ones(4))
        self.direction2idx = {'row': 0, 'col': 1, 'diag': 2, 'global': 3}

        # 对角索引缓存：键=(H,W)，值=(diag_idx, pos_idx)
        self._diag_cache: dict[tuple[int,int],
                               tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # 内部工具：生成 / 复用 对角线索引
    # ------------------------------------------------------------------ #
    def _diag_idx(self, H: int, W: int, device: torch.device):
        """
        返回 (flat_idx, pos_idx)，均在 `device` 上：
          flat_idx : 展平索引 (shape=[H*W])
          pos_idx  : 每个像素在其所在对角线序列中的位置 (shape=[H*W])
        只把 **CPU 版** 缓存起来，多卡时避免跨设备冲突。
        """
        key = (H, W)
        if key not in self._diag_cache:
            i = torch.arange(H).view(H, 1).expand(H, W)        # [H,W]
            j = torch.arange(W).view(1, W).expand(H, W)        # [H,W]
            diag = (j - i + (H - 1)).reshape(-1).long()        # 主对角线编号
            pos  = torch.where(j >= i, i, j).reshape(-1).long()# 在该对角线中的偏序
            self._diag_cache[key] = (diag, pos)                # 缓存在 CPU

        diag_cpu, pos_cpu = self._diag_cache[key]
        # 每次调用时搬到当前 GPU
        return diag_cpu.to(device, non_blocking=True), pos_cpu.to(device, non_blocking=True)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(self, ll, hf, edge=None):
        """
        ll : 低频 (B, C_l, H, W)
        hf : 高频 (B, C_h, H, W)
        """
        B, C_l, H, W = ll.shape
        _, C_h, _, _ = hf.shape
        d   = self.qh.out_channels
        N   = H * W
        scl = d ** -0.5                   # scaled dot-product 缩放

        # ------- Q K V 映射 ------
        if edge is None:
            Qh = self.qh(hf).view(B, d, N)
        else:
            # edge : [B,1,H,W] → 用它当 Query（扩到通道数）
            edge.detach()
            Qh = self.qh(edge.expand(-1, hf.size(1), -1, -1)).contiguous().view(B, d, N)
        Kh = self.kh(hf).view(B, d, N)
        Vh = self.vh(hf).view(B, d, N)
        Ql = self.ql(ll).view(B, d, N)
        Kl = self.kl(ll).view(B, d, N)
        Vl = self.vl(ll).view(B, d, N)

        # 空占位 (方向没启用时保持 0)
        z_ll = lambda: torch.zeros_like(ll)
        z_hf = lambda: torch.zeros_like(hf)

        # 归一化方向权重
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

            # scatter 回 (B,d,N)
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

        # ---------------- 残差 & 辅助卷积 --------------- #
        ll_out = self.aux_l(ll + sum(out_ll_list))
        hf_out = self.aux_h(hf + sum(out_hf_list))
        return ll_out, hf_out

class PostHead(nn.Module):
    def __init__(self, mid=32):
        super().__init__()
        self.guidance = nn.Sequential(
            nn.Conv2d(3, 8, 1, bias=False),  # RGB guidance
            nn.SiLU(True),
        )
        self.process = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid//4, bias=False),
        )
        self.fuse = nn.Conv2d(mid + 8, 1, 1, bias=False)

    def forward(self, alpha, rgb):
        g  = self.guidance(rgb)
        p  = self.process(alpha)
        out = self.fuse(torch.cat([p, g], 1))
        return torch.sigmoid(out)

class PostHeadResidual(nn.Module):
    def __init__(self, feat_ch, rgb_ch, mid_ch=128, use_rgb=True):
        super().__init__()

        self.use_rgb = use_rgb
        self.feat_ch = feat_ch
        self.mid_ch = mid_ch

        # 1. 投影层
        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch, mid_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # 2. 深度残差堆栈
        self.resblocks = nn.Sequential(
            ResidualBlock(mid_ch, mid_ch),
        )
        # 3. 再次激活扩展
        self.post = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # 4. 融合 & 细节残差预测
        if use_rgb:
            self.fuse = nn.Conv2d(mid_ch + rgb_ch, feat_ch, 1, bias=True)
        else:
            self.fuse = nn.Conv2d(mid_ch, feat_ch, 1, bias=True)
        # 5. 可导映射到 [0,1]
        self.act_out = nn.Hardsigmoid()

    def forward(self, recon_raw, rgb=None):
        x = self.proj(recon_raw)
        x = self.resblocks(x)
        x = self.post(x)
        if self.use_rgb and rgb is not None:
            x = torch.cat([x, rgb], dim=1)
        delta = self.fuse(x)
        # 用 Hardsigmoid 保障在 [0,1] 且梯度友好
        return self.act_out(recon_raw + delta)

class EnhancedPostHeadResidual(nn.Module):
    def __init__(self, feat_ch, rgb_ch=3, mid_ch=128, use_rgb=True):
        super().__init__()
        self.use_rgb = use_rgb

        # 1. 初始投影：将 recon_raw 映射到中间通道
        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 2. Local 分支：提取局部结构特征（group conv 保留空间纹理）
        self.local_branch = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch // 4, bias=False)

        # 3. Global 分支：通道注意力（类似 squeeze-excitation）
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 4. 正宗 ResidualBlock（复用你的已有定义）
        self.resblock = ResidualBlock(mid_ch, mid_ch)

        # 5. 融合 & 输出 delta
        in_fuse_ch = mid_ch + rgb_ch if use_rgb else mid_ch
        self.fuse = nn.Conv2d(in_fuse_ch, feat_ch, kernel_size=1, bias=True)

        # 6. Hardsigmoid 保障输出在 [0,1]
        self.act_out = nn.Hardsigmoid()

    def forward(self, recon_raw, rgb=None):
        x = self.proj(recon_raw)  # B, mid_ch, H, W

        # Local-global 注意力调制
        local_feat = self.local_branch(x)                   # 局部响应
        global_weight = self.global_branch(x)               # 通道注意力
        fused_feat = local_feat * global_weight + local_feat  # 残差调制

        # 经过 ResidualBlock（包含 skip connection）
        res_feat = self.resblock(fused_feat)

        # 可选 RGB 引导
        if self.use_rgb and rgb is not None:
            res_feat = torch.cat([res_feat, rgb], dim=1)

        delta = self.fuse(res_feat)
        return self.act_out(recon_raw + delta)  # 明确 residual


class ShallowDilatedStem(nn.Module):
    """保持 H×W 的 dilated conv 路径，用来保留 1–2 px 级细节"""
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

# -----------------------------------------------------------------------------
# 整体网络接口更新：两个分支编码器
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

        # ll_in_ch = in_channels + in_channels * L          # = 3 + 3L
        # hf_in_ch = in_channels + 3 * in_channels * L      # = 3 + 9L

        # if no rgb involved at all
        ll_in_ch = in_channels * L
        hf_in_ch = 3 * in_channels * L 

        # 空间
        self.sa_ll = nn.Sequential(
            nn.Conv2d(in_channels * L, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sa_hf = nn.Sequential(
            nn.Conv2d(3*in_channels*L, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # 通道
        # self.se_hf = HFSE(C_in = 3*in_channels*L, r=8)
        self.se_ll = LFSE(C_in=in_channels * len(wavelet_list), r=8)
        self.prefuse_hf = RGDF(channels=3*in_channels*L // 3, use_edge=True)

        self.stem_shallow = ShallowDilatedStem(
            hf_in_ch, base_channels,
        )

        # ── 编码器 ──
        self.encoder_ll = BranchEncoder(ll_in_ch, base_channels,
                                        num_layers, blocks_per_layer)
        self.encoder_hf = BranchEncoder(hf_in_ch, base_channels,
                                        num_layers, blocks_per_layer)

        encoder_channels = [base_channels * (2 ** i)      # [C,2C,4C,8C]
                            for i in range(num_layers)]

        self.cross_attn_bn = CrossAxisAttention(
            C_l=encoder_channels[-1],     # 最深层通道
            C_h=encoder_channels[-1],
            d_head=d_head,
            modes=modes_bn,
            )
        self.cross_attn_mid = CrossAxisAttention(
            C_l=encoder_channels[-2],     # 倒数第二层
            C_h=encoder_channels[-2],
            d_head=d_head//2,
            modes=modes_mid,
            )

        # （可选）HF 通道增强
        """hf_enhance_structure = build_hf_enhance_structure(
            num_layers, layers_to_enhance=(1, 0), factor=2)
        self.hf_expanders = nn.ModuleDict({
            str(i): nn.Conv2d(encoder_channels[i],
                              encoder_channels[i] * f, 1, bias=False)
            for i, f in hf_enhance_structure.items()
        })"""

        # ── 解码器 ──
        self.decoder_ll = BranchDecoder(encoder_channels, base_channels, 1 * L)
        self.decoder_hf = BranchDecoder(encoder_channels, base_channels,
                                        3 * 1 * L, enhance_structure=None)
        self.shallow2ll = nn.Conv2d(base_channels, 1*L, 1, bias=False)
        self.shallow2hf = nn.Conv2d(base_channels, 3*1*L, 1, bias=False)

        # 对 BN‐level cross‐attn 输出做投影
        self.attn_bn_head_ll  = nn.Conv2d(encoder_channels[-1], 1, kernel_size=1, bias=False)
        self.attn_bn_head_hf  = nn.Conv2d(encoder_channels[-1], 1, kernel_size=1, bias=False)
        # 对 Mid‐level cross‐attn 输出做投影 
        self.attn_mid_head_ll = nn.Conv2d(encoder_channels[-2], 1, kernel_size=1, bias=False)
        self.attn_mid_head_hf = nn.Conv2d(encoder_channels[-2], 1, kernel_size=1, bias=False)
        # 对 Shallow 分支特征做投影
        self.shallow_head     = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)

        # ── post fuse ──
        """self.post = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),   # 1→8, dense conv
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False),  # DW conv
            nn.Conv2d(8, 1, 3, padding=1, bias=False),    # fuse back to single‑channel α
            nn.Sigmoid()
        )"""
        """self.post = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),   # 1→8, dense conv
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, groups=4, bias=False),  # DW conv
            nn.Conv2d(8, 1, 3, padding=1, bias=False),    # fuse back to single‑channel α
            nn.Sigmoid()
        )"""
        # self.post = PostHead(mid=32)
        """self.post = PostHeadResidual(
            feat_ch=1, rgb_ch=3, mid_ch=64, use_rgb=False
        )"""
        self.post = EnhancedPostHeadResidual(
            feat_ch=1, rgb_ch=3, mid_ch=64, use_rgb=False
        )


    # -----------------------------------------------------------------
    def forward(self, rgb: torch.Tensor, guide_mask: torch.Tensor):
        # kernel=5, 轻微锐化，alpha 可调（0.5~1.0 之间）
        alpha = 0.7
        blur  = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2)  # 均值模糊
        rgb_sharp = rgb + alpha * (rgb - blur)

        # ---------- Wavelet 分解 ----------
        ll_cat, hf_cat = self.wave.multi_dwt_concat(rgb_sharp)
        edge_map = hf_cat.abs().mean(1, keepdim=True)

        ll_cat = self.se_ll(ll_cat)
        ll_cat = ll_cat * (1.0 + self.sa_ll(ll_cat))

        hf_cat = self.prefuse_hf(hf_cat, edge_map)  
        hf_cat = hf_cat * (1.0 + self.sa_hf(hf_cat))

        H1, W1 = ll_cat.shape[-2:]
        rgb_ds = F.interpolate(rgb_sharp, size=(H1, W1),
                               mode='bilinear', align_corners=False)

        # in_ll = torch.cat([rgb_ds, ll_cat], dim=1)
        # in_hf = torch.cat([rgb_ds, hf_cat], dim=1)

        # if no rgb involved
        in_ll = ll_cat
        in_hf = hf_cat


        feats_ll = self.encoder_ll(in_ll)   # list 长度 = num_layers
        feats_hf = self.encoder_hf(in_hf)

        # ——— BN‐level cross‐attn ———
        edge_bn = F.interpolate(edge_map, size=feats_ll[-1].shape[-2:], mode='nearest')
        attn_bn_ll, attn_bn_hf = self.cross_attn_bn(feats_ll[-1], feats_hf[-1], edge_bn)
        feats_ll[-1], feats_hf[-1] = attn_bn_ll, attn_bn_hf
        pred_attn_bn_ll = self.attn_bn_head_ll(attn_bn_ll)
        pred_attn_bn_hf = self.attn_bn_head_hf(attn_bn_hf)

        edge_mid = F.interpolate(edge_map, size=feats_ll[-2].shape[-2:], mode='nearest')
        attn_mid_ll, attn_mid_hf = self.cross_attn_mid(feats_ll[-2], feats_hf[-2], edge_mid)
        feats_ll[-2], feats_hf[-2] = attn_mid_ll, attn_mid_hf
        pred_attn_mid_ll = self.attn_mid_head_ll(attn_mid_ll)
        pred_attn_mid_hf = self.attn_mid_head_hf(attn_mid_hf)

        # ---------- HF 通道扩展 ----------
        """for idx, exp in self.hf_expanders.items():
            feats_hf[int(idx)] = exp(feats_hf[int(idx)])"""

        # ---------- 解码 & IWT 重建 ----------
        dec_feats_ll, dec_ll = self.decoder_ll(feats_ll)      # dec_ll: 360×640
        dec_feats_hf, dec_hf = self.decoder_hf(feats_hf)

        hf_cat_up   = F.interpolate(hf_cat, size=rgb_sharp.shape[-2:], mode='nearest')
        # shallow_feat = self.stem_shallow(torch.cat([rgb_sharp, hf_cat_up], 1))  # 720×1280
        # if no rgn involved
        shallow_feat = self.stem_shallow(hf_cat_up)
        # ↓↓↓  关键：下采样到 decoder 最浅层 / ll_cat 分辨率 (H/2, W/2)
        shallow_ds = F.interpolate(
            shallow_feat, size=dec_ll.shape[-2:],  # (360,640)
            mode='bilinear', align_corners=False)
        # 生成单通道 shallow map 预测
        pred_shallow   = self.shallow_head(shallow_ds)

        # 1) 给 Decoder feature 注入细节
        # dec_feats_ll[-1] = dec_feats_ll[-1] + shallow_ds
        # dec_feats_hf[-1] = dec_feats_hf[-1] + shallow_ds

        # 2) 注入 IWT 重建分量
        """proj_ll = self.shallow2ll(shallow_ds)
        proj_hf = self.shallow2hf(shallow_ds)
        dec_ll = dec_ll + proj_ll
        dec_hf = dec_hf + proj_hf"""

        recon_raw = self.wave.multi_iwt_from_concat(dec_ll, dec_hf)
        recon     = self.post(recon_raw, None)
        """with torch.no_grad():
            r = recon.view(-1)
            print(f"Sigmoid→recon range: min={r.min().item():.4f}, max={r.max().item():.4f}")
            print(f"  mean={r.mean().item():.4f}, std={r.std().item():.4f}")
            pct_in = ((r>=0)&(r<=1)).float().mean().item()*100
            print(f"  percent in [0,1]: {pct_in:.2f}%")
        """
        return (feats_ll, feats_hf, ll_cat, hf_cat,
                dec_feats_ll, dec_feats_hf, pred_shallow, 
                pred_attn_bn_ll, pred_attn_bn_hf, pred_attn_mid_ll, 
                pred_attn_mid_hf, recon_raw, recon)



if __name__ == "__main__":
    # 测试 main 函数
    # 使用默认配置：wavelet_list=["db1","db2","db4"], base_channels=64, num_layers=4, blocks_per_layer=2
    model = refiner_xnet()
    # 构造假输入：batch=1, 3通道, 高720, 宽1280
    dummy_input = torch.randn(2, 3, 720, 1280)
    dummy_guide = torch.randn(2, 1, 720, 1280)
    feats_ll, feats_hf, ll_cat, hf_cat, dec_feats_ll, dec_feats_hf, \
    shallow_ds, attn_bn_ll, attn_bn_hf, attn_mid_ll, \
    attn_mid_hf, recon_raw, recon = model(dummy_input, dummy_guide)
    # 打印各分支特征和重构图形状
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