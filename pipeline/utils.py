import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unknown_mask(trimap: torch.Tensor) -> torch.Tensor:
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
    # unknown region to 1, other to 0
    return ((t > 0.01) & (t < 0.99)).float()


def save_visualization(
    rgb, init_mask, gt, trimap,
    ll_cat, hf_cat,
    feats_ll, feats_hf,
    dec_feats_ll, dec_feats_hf,
    recon, recon_raw,
    pred_shallow,
    pred_attn_bn_ll, pred_attn_bn_hf,
    pred_attn_mid_ll, pred_attn_mid_hf,
    wnames,
    save_path,
    up_mode: str = "bilinear",
):
    """
    visualize the full refiner_xnet pipeline (generated to save_path)
    supports arbitrary wavelet combination count L = len(wnames), expanded view of all
    """
    # ---------------- basic info ---------------- #
    B, _, H, W = rgb.shape
    _, ll_ch, H1, W1 = ll_cat.shape
    L = len(wnames)  # wavelet levels
    C = ll_ch // L  # number of low-frequency channels for each wavelet
    cols = max(4, L + 1)  # min 4 cols; if L>3 -> L+1 (+1 for recon/error)

    # ---------------- denorm (imagenet) ---------------- #
    def denorm(x: torch.Tensor) -> torch.Tensor:
        """x∈[-mean/std, (1-mean)/std] → [0,1]"""
        mean = IMAGENET_MEAN.to(x.device)
        std  = IMAGENET_STD.to(x.device)
        return (x * std + mean).clamp(0, 1)

    rgb_vis = denorm(rgb)[0].permute(1, 2, 0).cpu().numpy()
    init_vis = init_mask[0, 0].cpu().numpy()
    gt_vis = gt[0, 0].cpu().numpy()
    tri_vis = trimap[0, 0].cpu().numpy()
    recon_vis = recon[0, 0].detach().cpu().numpy()
    err_vis = np.abs(gt_vis - recon_vis)

    kernels = torch.tensor([
        [[-0.5,  0.5], [-0.5,  0.5]],  # LH
        [[-0.5, -0.5], [ 0.5,  0.5]],  # HL
        [[ 0.5, -0.5], [-0.5,  0.5]],  # HH
    ], dtype=torch.float32).unsqueeze(1).to(gt.device)
    with torch.no_grad():
        band_tensor = conv_dwt_energy_mask(
            gt.to(torch.float32),        # make sure is float32
            kernels,
            threshold_quantile=0.75,
            dilation_radius=5,
        )
        # edge as skeleton (can be adjusted if needed)
        edge_tensor = (band_tensor > 0).float()
        band_np = band_tensor[0,0].cpu().numpy()
        edge_np = edge_tensor[0,0].cpu().numpy()
        overlay = np.stack([band_np, band_np, band_np], axis=-1)
        overlay[edge_np > 0] = [1.0, 0.0, 0.0]

    ll_list = ll_cat.split(C, dim=1)
    hf_list = hf_cat.split(3 * C, dim=1)

    # ---------------- dynamic row calculation ---------------- #
    total_enc = max(len(feats_ll), len(feats_hf))
    enc_rows = (total_enc + 3) // 4
    total_dec = max(len(dec_feats_ll), len(dec_feats_hf))
    dec_rows = (total_dec + 3) // 4
    total_rows = 4 + enc_rows * 2 + dec_rows * 2

    fig, axs = plt.subplots(
        total_rows, cols, figsize=(4 * cols, 3 * total_rows)
    )

    # ========= line 1：RGB / Init / GT / Trimap ========= #
    first_row_imgs = [rgb_vis, init_vis, gt_vis, overlay]
    first_row_titles = ["RGB", "Init Mask", "GT", "DWT Loss Region"]
    for i, (img, ttl) in enumerate(zip(first_row_imgs, first_row_titles)):
        axs[0, i].imshow(img if i == 0 else img, cmap=None if i == 0 else "gray")
        axs[0, i].set_title(ttl)
    for j in range(len(first_row_imgs), cols):
        axs[0, j].axis("off")

    # ========= line 2：wavelet LL + Recon ========= #
    for i in range(L):
        ll_up = F.interpolate(
            ll_list[i][:, :1].float(), size=(H, W), mode=up_mode
        )[0, 0].cpu()
        axs[1, i].imshow(ll_up, cmap="gray")
        axs[1, i].set_title(f"{wnames[i]} LL")
    axs[1, cols - 2].imshow(recon_raw[0,0], cmap="gray")
    axs[1, cols - 2].set_title("Recon (Raw)")
    axs[1, cols - 1].imshow(recon_vis, cmap="gray")
    axs[1, cols - 1].set_title("Pred")
    for j in range(L, cols - 1):
        axs[1, j].axis("off")

    # ========= line 3：wavelet HF + ErrMap ========= #
    for i in range(L):
        # expand to (B, C, 3, H/2, W/2)
        hf_i = hf_list[i].view(B, C, 3, H1, W1).float()

        # compute energy map instead of simple average
        # +1e-12 for numerical stability to avoid sqrt(0)
        hf_energy = torch.sqrt((hf_i ** 2).sum(dim=2) + 1e-12)   # -> (B,C,H/2,W/2)

        # use first channel for visualization and upsample to original size
        hf_up = F.interpolate(hf_energy[:, :1], size=(H, W), mode=up_mode)[0, 0]

        # adaptive vmax to avoid extreme edges darkening the visualization
        vmax = hf_up.abs().quantile(0.995).item() + 1e-8
        axs[2, i].imshow(hf_up.cpu(), cmap="bwr", vmin=-vmax, vmax=vmax)
        axs[2, i].set_title(f"{wnames[i]} HF")
    axs[2, cols - 1].imshow(err_vis, cmap="hot")
    axs[2, cols - 1].set_title("GT – Recon Err")
    for j in range(L, cols - 1):
        axs[2, j].axis("off")

    # ========= line 4：Auxiliary Heads ========= #
    aux_row = 3
    titles = ["Att BN LL", "Attn BN HF", "Attn Mid LL", "Attn Mid HF"]
    images = [pred_attn_bn_ll, pred_attn_bn_hf, pred_attn_mid_ll, pred_attn_mid_hf]
    for i in range(len(titles)):
        m = F.interpolate(images[i].float(), size=(H, W), mode=up_mode)[0, 0].cpu().numpy()
        axs[aux_row, i].imshow(m, cmap="gray")
        axs[aux_row, i].set_title(titles[i])
    for j in range(len(images), cols):
        axs[aux_row, j].axis("off")

    # ========= Encoder ========= #
    base_row = 4
    for idx in range(total_enc):
        r_ll = base_row + (idx // 4) * 2
        r_hf = r_ll + 1
        c    = idx % 4
        # LL
        if idx < len(feats_ll):
            fmap = feats_ll[idx].mean(1)[0]
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_ll, c].imshow(norm.cpu(), cmap="viridis")
            axs[r_ll, c].set_title(f"Enc LL L{idx}")
        else:
            axs[r_ll, c].axis("off")
        # HF
        if idx < len(feats_hf):
            fmap = feats_hf[idx].mean(1)[0]
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_hf, c].imshow(norm.cpu(), cmap="viridis")
            axs[r_hf, c].set_title(f"Enc HF L{idx}")
        else:
            axs[r_hf, c].axis("off")
        for j in range(4, cols):
            axs[r_ll, j].axis("off")
            axs[r_hf, j].axis("off")

    # ========= Decoder ========= #
    dec_base = base_row + enc_rows * 2
    for idx in range(total_dec):
        r_ll = dec_base + (idx // 4) * 2
        r_hf = r_ll + 1
        c    = idx % 4
        # LL
        if idx < len(dec_feats_ll):
            fmap = dec_feats_ll[idx][0].mean(0)
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_ll, c].imshow(norm.cpu(), cmap="magma")
            axs[r_ll, c].set_title(f"Dec LL L{idx}")
        else:
            axs[r_ll, c].axis("off")
        # HF
        if idx < len(dec_feats_hf):
            fmap = dec_feats_hf[idx][0].mean(0)
            norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axs[r_hf, c].imshow(norm.cpu(), cmap="magma")
            axs[r_hf, c].set_title(f"Dec HF L{idx}")
        else:
            axs[r_hf, c].axis("off")
        for j in range(4, cols):
            axs[r_ll, j].axis("off")
            axs[r_hf, j].axis("off")

    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def conv_dwt_energy_mask(
    alpha_gt: torch.Tensor,
    kernels: torch.Tensor,  # (3,1,2,2) - Haar high-pass kernels
    threshold_quantile: float = 0.90,
    dilation_radius: int = 2,
) -> torch.Tensor:
    """
    Compute an edge band mask using convolution-based DWT HF energy.
    1) Compute HF with Haar high-pass kernels;
    2) Take energy = sqrt(sum of squares) of HF;
    3) Binarize the energy map using a quantile threshold;
    4) Dilate to obtain the band mask.

    Returns:
        band: (B, 1, H, W) float in {0, 1}
    """
    alpha = alpha_gt.float()
    hp = F.conv2d(alpha, kernels.to(alpha.device), padding=1)  # (B,3,H,W)
    energy = torch.sqrt((hp ** 2).sum(dim=1, keepdim=False) + 1e-6)  # (B,H,W)
    energy = energy.unsqueeze(1)  # (B,1,H,W)

    # 按 batch 求 quantile
    B = alpha.size(0)
    band_masks = []
    for b in range(B):
        e_b = energy[b]
        th = torch.quantile(e_b.view(-1), threshold_quantile)
        mask = (e_b > th).float()
        # 膨胀
        k = 2 * dilation_radius + 1
        band = F.max_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=dilation_radius)
        band_masks.append(band)

    return torch.cat(band_masks, dim=0)  # (B,1,H,W)


def sobel_edge_weight(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradient magnitude normalized to [0,1] for alpha maps.
    """
    # Create Sobel kernels with same dtype as alpha
    kx = torch.tensor([[-1,0,1], [-2,0,2], [-1,0,1]],
                      device=alpha.device,
                      dtype=alpha.dtype).view(1,1,3,3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(alpha, kx, padding=1)
    gy = F.conv2d(alpha, ky, padding=1)
    grad = torch.sqrt(gx*gx + gy*gy)
    maxv = grad.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    return (grad / maxv).clamp(0, 1)


# ------------------------------------------------------------------ #
# batch change BN to GN for stablizing
# ------------------------------------------------------------------ #
def pick_gn_groups(C: int) -> int:
    # prioritize 32/16/8/4/2/1 to ensure divisibility of channel count
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1


def bn_to_gn(module: nn.Module):
    for name, m in list(module.named_children()):
        if isinstance(m, nn.BatchNorm2d):
            C = m.num_features
            gn = nn.GroupNorm(
                num_groups=pick_gn_groups(C), num_channels=C,
                affine=True, eps=1e-5
            )
            setattr(module, name, gn)
        else:
            bn_to_gn(m)


def count_bn(module):
    return sum(isinstance(m, nn.BatchNorm2d) for m in module.modules())


def count_gn(module):
    return sum(isinstance(m, nn.GroupNorm) for m in module.modules())



