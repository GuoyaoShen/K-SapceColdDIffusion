import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import fastmri

import glob
import os
from PIL import Image
from net.u_net_diffusion import cycle, EMA, loss_backwards
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor
import os
import errno
from collections import OrderedDict



class Visualizer_Kspace_ColdDiffusion(object):
    def __init__(
            self,
            diffusion_model,
            *,
            ema_decay=0.995,
            dataloader_test=None,
            load_path=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.dl_test = dataloader_test

        self.reset_parameters()
        self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def show_intermediate_kspace_cold_diffusion(self, t, idx_case):
        for i_case, data in enumerate(self.dl_test):
            if i_case != idx_case:
                continue
            kspace, mask, mask_fold = data  # [B,Nc(1),H,W,2]
            kspace = kspace.cuda()
            mask = mask.cuda()
            mask_fold = mask_fold.cuda()
            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

            # [B,Nc,H,W,2]
            xt, direct_recons, sample_imgs = self.ema_model.sample(kspace, mask, mask_fold, t=t)
            kspacet = fastmri.fft2c(xt)

            gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
            direct_recons_abs = fastmri.complex_abs(direct_recons)  # [B,Nc,H,W]
            sample_imgs_abs = fastmri.complex_abs(sample_imgs)  # [B,Nc,H,W]

            xt = xt[0, 0]  # [H,W,2]
            kspacet = kspacet[0, 0]
            kspace = kspace[0, 0]
            gt_imgs_abs = gt_imgs_abs[0, 0]  # [H,W]
            direct_recons_abs = direct_recons_abs[0, 0]
            sample_imgs_abs = sample_imgs_abs[0, 0]

            return xt, kspacet, gt_imgs_abs, direct_recons_abs, sample_imgs_abs, kspace


def show_intermediate_kspace_Gmask(dl_test, idx_case, t, t_all):
    '''
    Show intermediate kspace mask given a time step number.
    '''
    for i_case, data in enumerate(dl_test):
        if i_case != idx_case:
            continue
        kspace, mask, mask_fold = data  # [B,Nc(1),H,W,2]
        B, Nc, H, W, C = kspace.shape
        gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]

        mask = mask_fold[0, 0]  # [B,1,H,W] to [H,W]
        mask_t = mask.clone()

        # mask_inv = torch.ones_like(mask) - mask
        mask_inv_idx = torch.argwhere(abs(mask) < 1e-6)
        mask_inv_idx = mask_inv_idx[torch.randperm(mask_inv_idx.shape[0])]
        len_keep = int(np.floor((t_all - t) / t_all * mask_inv_idx.shape[0]))
        mask_inv_idx_keep = mask_inv_idx[:len_keep]

        for _, data in enumerate(mask_inv_idx_keep):
            [ix, iy] = data
            mask_t[ix, iy] = 1

        mask = mask[None, None, ...]
        mask_t = mask_t[None, None, ...]
        up_sample = nn.Upsample(scale_factor=(kspace.shape[-2]/mask_fold.shape[-1]), mode='nearest')
        mask = up_sample(mask)
        mask_t = up_sample(mask_t)

        # get image space
        mask_t = mask_t[..., None]
        kspace_t = mask_t * kspace
        imgs_t = fastmri.ifft2c(kspace_t)

        gt_imgs_abs = fastmri.complex_abs(gt_imgs)  # [B,Nc,H,W]
        imgs_t_abs = fastmri.complex_abs(imgs_t)  # [B,Nc,H,W]
        mask_t = mask_t[..., 0]

        return mask_t, mask, imgs_t_abs, gt_imgs_abs


def recon_unet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
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

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_unet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
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

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg


def recon_wnet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_wnet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg


def recon_varnet(
        dataloader,
        net,
        device,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    zf_list = []
    tg_list = []
    recon_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf_list.append(X.detach().cpu().squeeze(1))
            tg_list.append(tg.cpu())
            recon_list.append(pred.cpu())
    return recon_list, zf_list, tg_list


def recon_slice_varnet(
        dataloader,
        net,
        device,
        idx_case,
):
    '''
    Reconstruct image from the dataloader.
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            zf = X.detach().cpu().squeeze(1)
            tg = tg.cpu()
            pred = pred.cpu()
            break
    return pred, zf, tg

