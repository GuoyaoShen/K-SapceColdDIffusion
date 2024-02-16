import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.evaluation_utils import *



def recon_unet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = X.detach().squeeze(1)
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()


def recon_kspace_cold_diffusion(
        dataloader,
        net,
        timesteps,
        device,
        idx_case,
        show_info=True,
):
    assert isinstance(net, KspaceDiffusion), "Input net must be a KspaceDiffusion."
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            kspace, mask, mask_fold = data  # [B,Nc,H,W,2]
            kspace = kspace.to(device)
            mask = mask.to(device)
            mask_fold = mask_fold.to(device)
            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

            # network forward
            xt, direct_recons, sample_imgs = net.sample(kspace, mask, mask_fold, t=timesteps)
            gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
            direct_recons_abs = fastmri.complex_abs(direct_recons)  # [B,Nc,H,W]
            sample_imgs_abs = fastmri.complex_abs(sample_imgs)  # [B,Nc,H,W]
            # combine coil
            gt_imgs_abs = fastmri.rss(gt_imgs_abs, dim=1)  # [B,H,W]
            direct_recons_abs = fastmri.rss(direct_recons_abs, dim=1)
            sample_imgs_abs = fastmri.rss(sample_imgs_abs, dim=1)

            # evaluation metrics
            tg = gt_imgs_abs.detach()  # [B,H,W]
            pred_dir = direct_recons_abs.detach()
            pred = sample_imgs_abs.detach()
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            i_nmse_dir = calc_nmse_tensor(tg, pred_dir)
            i_psnr_dir = calc_psnr_tensor(tg, pred_dir)
            i_ssim_dir = calc_ssim_tensor(tg, pred_dir)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))
                print('Direct Recon NMSE: ' + str(i_nmse_dir) + '|| PSNR: ' + str(i_psnr_dir) + '|| SSIM: ' + str(i_ssim_dir))

            break
    return pred.cpu().numpy(), tg.cpu().numpy(), pred_dir.cpu().numpy()
