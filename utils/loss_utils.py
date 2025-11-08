#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def depth_loss(pred_depth, depth_gt, mask_gt, alpha=None, w_alpha=0.0,
               robust='charbonnier', clip=None, log_depth=False):
    """
    pred_depth: (H,W) torch, same unit/coordinate as depth_gt
    depth_gt:   (H,W) torch
    mask_gt:    (H,W) torch.bool
    alpha:      (H,W) torch in [0,1], optional, use for ignoring unvisible regions
    w_alpha:    alpha weight, 0 means no alpha
    robust:     'charbonnier' or 'huber'
    clip:       optional, rip off large residuals, for example (0.1, 80.0)
    log_depth:  while 'True', comparing in log space(scales be more stable)
    """

    if clip is not None:
        d0, d1 = clip
        pred_depth = pred_depth.clamp(d0, d1)
        depth_gt   = depth_gt.clamp(d0, d1)

    if log_depth:
        # avoid log(0), only in valid mask + positive depth
        eps = 1e-6
        pred_depth = torch.log(pred_depth.clamp_min(eps))
        depth_gt   = torch.log(depth_gt.clamp_min(eps))

    resid = pred_depth - depth_gt  # (H,W)
    if robust == 'charbonnier':
        perpx = charbonnier(resid)
    elif robust == 'huber':
        perpx = F.huber_loss(pred_depth, depth_gt, reduction='none', delta=0.1)
    else:
        perpx = resid.abs()

    # merge mask（visible LiDAR + optional alpha）
    if alpha is not None and w_alpha > 0:
        weight = mask_gt.float() * (1.0 - w_alpha + w_alpha * alpha)
    else:
        weight = mask_gt.float()

    loss = (perpx * weight).sum() / (weight.sum() + 1e-8)
    return loss