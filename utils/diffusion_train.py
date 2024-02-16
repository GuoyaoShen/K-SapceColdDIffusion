# trainer class
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


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('.module', '')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


class Trainer(object):
    """
       diffusion trainer

        Args:
        ----------
        diffusion_model : model
            diffusion model
        ema_decay : float
            exponential mean average dacay.
        image_size : int
            image size of H W
        train_batch_size : int
            batch size for training
        train_lr: float,
            learning rate for training .
        train_num_steps: int
            num of training times steps for diffusion process
        gradient_accumulate_every: int
            gradient update for each # time step
        fp16 : bool
            if using 16 float
        step_start_ema : int
            step to start update by ema
        update_ema_every: int,
            ema update for each # train step
        results_folder: string,
            result save folder
        load_path: string,
            model load folder
       dataloader_train: dataloader module,
            dataloader for training
        dataloader_test: dataloader module,
            dataloader for tes testing
        """

    def __init__(
            self,
            diffusion_model,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=200,
            results_folder='./results',
            load_path=None,
            dataloader_train=None,
            dataloader_test=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.dl = cycle(dataloader_train)
        self.dataloader_test = dataloader_test
        self.dl_test = cycle(dataloader_test)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)
        self.nmse = 0
        self.psnr = 0
        self.ssim = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def train(self):

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        pbar = tqdm(range(self.train_num_steps), desc='LOSS')
        # while self.step < self.train_num_steps:
        for step in pbar:
            self.step = step
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                # data = next(self.dl)
                kspace, mask, mask_fold = next(self.dl)
                kspace = kspace.cuda()
                mask = mask.cuda()
                mask_fold = mask_fold.cuda()

                loss = torch.mean(self.model(kspace, mask, mask_fold))  # change for DP
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            pbar.set_description("Loss=%f" % (u_loss / self.gradient_accumulate_every))

            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)


            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:

                acc_loss = acc_loss / (self.save_and_sample_every + 1)
                print(f'Mean LOSS of last {self.step}: {acc_loss}')
                acc_loss = 0

                self.save(self.step)
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)
        self.save(self.step+1)
        print('training completed')

    def test(self, t, num_samples=1):
        torch.set_grad_enabled(False)
        sample_imgs_list = []
        gt_imgs_list = []
        xt_list = []
        direct_recons_list = []

        nmse = 0
        psnr = 0
        ssim = 0

        print('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        with torch.no_grad():
            pbar = tqdm(range(len(self.dataloader_test)), desc='LOSS')
            for idx in pbar:
                kspace, mask, mask_fold = next(self.dl_test) # [B,Nc,H,W,2]
                kspace = kspace.cuda()
                mask = mask.cuda()
                mask_fold = mask_fold.cuda()
                B, Nc, H, W, C = kspace.shape
                gt_imgs = fastmri.ifft2c(kspace) # [B,Nc,H,W,2]

                # [B,Nc,H,W,2]
                if num_samples == 1:
                    xt, direct_recons, sample_imgs = self.ema_model.sample(kspace, mask, mask_fold, t=t)
                    if idx == 0:
                        print('direct_recons.shape:', direct_recons.shape)
                else:
                    for i_sample in range(num_samples):
                        xti, direct_reconsi, sample_imgsi = self.ema_model.sample(kspace, mask, mask_fold, t=t)
                        if i_sample == 0:
                            if idx == 0:
                                print('direct_reconsi.shape:', direct_reconsi.shape)
                            xt = xti
                            direct_recons = direct_reconsi
                            sample_imgs = sample_imgsi
                        else:
                            xt = xti
                            direct_recons = torch.cat((direct_recons, direct_reconsi), dim=1)
                            sample_imgs = torch.cat((sample_imgs, sample_imgsi), dim=1)
                    #         direct_recons += direct_reconsi
                    #         sample_imgs += sample_imgsi
                    # xt /= num_samples
                    # direct_recons /= num_samples
                    # sample_imgs /= num_samples

                gt_imgs_abs = fastmri.complex_abs(gt_imgs) # [B,Nc,H,W]
                sample_imgs_abs = fastmri.complex_abs(sample_imgs) # [B,Nc,H,W]
                # # combine coil
                # gt_imgs_abs = fastmri.rss(gt_imgs_abs, dim=1)  # [B,H,W]
                # sample_imgs_abs = fastmri.rss(sample_imgs_abs, dim=1)
                # combine samples
                gt_imgs_abs = torch.mean(gt_imgs_abs, dim=1)  # [B,H,W]
                sample_imgs_abs = torch.mean(sample_imgs_abs, dim=1)

                nmseb = 0
                psnrb = 0
                ssimb = 0
                for i in range(B):
                    nmseb += calc_nmse_tensor(gt_imgs_abs, sample_imgs_abs)
                    psnrb += calc_psnr_tensor(gt_imgs_abs, sample_imgs_abs)
                    ssimb += calc_ssim_tensor(gt_imgs_abs, sample_imgs_abs)
                nmseb /= B
                psnrb /= B
                ssimb /= B
                nmse += nmseb
                psnr += psnrb
                ssim += ssimb
                if idx == 0:
                    print('sample_imgs_abs.shape:', sample_imgs_abs.shape)
                    print('sample_imgs_abs slice shape:', gt_imgs_abs[0].unsqueeze(0).shape)
                    print('Batch PSNR:%.5f || SSIM:%.5f' % (psnrb, ssimb))

                sample_imgs_list.append(sample_imgs)
                gt_imgs_list.append(gt_imgs)
                xt_list.append(xt)
                direct_recons_list.append(direct_recons)

            nmse = nmse / len(self.dataloader_test)
            psnr = psnr / len(self.dataloader_test)
            ssim = ssim / len(self.dataloader_test)

            self.nmse = nmse
            self.psnr = psnr
            self.ssim = ssim
            print('### NMSE: ' + str(self.nmse) + '|| PSNR: ' + str(self.psnr) + '|| SSIM: ' + str(self.ssim))
            print('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)
        return sample_imgs_list, gt_imgs_list, xt_list, direct_recons_list

    def recon_slice(self, t, idx_case, num_samples=1):
        torch.set_grad_enabled(False)

        print('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        for idx, data in enumerate(self.dataloader_test):
            if idx != idx_case:
                continue
            kspace, mask, mask_fold = data # [B,Nc,H,W,2]
            kspace = kspace.cuda()
            mask = mask.cuda()
            mask_fold = mask_fold.cuda()
            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace) # [B,Nc,H,W,2]

            # [B,Nc,H,W,2]
            if num_samples == 1:
                xt, direct_recons, sample_imgs = self.ema_model.sample(kspace, mask, mask_fold, t=t)
                if idx == 0:
                    print('direct_recons.shape:', direct_recons.shape)
            else:
                for i_sample in range(num_samples):
                    xti, direct_reconsi, sample_imgsi = self.ema_model.sample(kspace, mask, mask_fold, t=t)
                    if i_sample == 0:
                        if idx == 0:
                            print('direct_reconsi.shape:', direct_reconsi.shape)
                        xt = xti
                        direct_recons = direct_reconsi
                        sample_imgs = sample_imgsi
                    else:
                        xt = xti
                        direct_recons = torch.cat((direct_recons, direct_reconsi), dim=1)
                        sample_imgs = torch.cat((sample_imgs, sample_imgsi), dim=1)

            gt_imgs_abs = fastmri.complex_abs(gt_imgs) # [B,Nc,H,W]
            sample_imgs_abs = fastmri.complex_abs(sample_imgs) # [B,Nc,H,W]
            # combine samples
            gt_imgs_abs = torch.mean(gt_imgs_abs, dim=1)  # [B,H,W]
            sample_imgs_abs = torch.mean(sample_imgs_abs, dim=1)

            nmseb = 0
            psnrb = 0
            ssimb = 0
            for i in range(B):
                nmseb += calc_nmse_tensor(gt_imgs_abs, sample_imgs_abs)
                psnrb += calc_psnr_tensor(gt_imgs_abs, sample_imgs_abs)
                ssimb += calc_ssim_tensor(gt_imgs_abs, sample_imgs_abs)
            nmseb /= B
            psnrb /= B
            ssimb /= B
            print('### NMSE: ' + str(nmseb) + '|| PSNR: ' + str(psnrb) + '|| SSIM: ' + str(ssimb))
            print('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)
            break
        return sample_imgs, gt_imgs, xt, direct_recons
