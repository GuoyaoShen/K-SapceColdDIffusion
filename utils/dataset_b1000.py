import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from help_func import print_var_detail

import fastmri



class DatasetB1000(Dataset):
    def __init__(
            self,
            path,
            mask_func,
            img_size=256,
            img_mode='B1000',
            net_mode='unet',
    ):
        '''
        path: path of the dataset.
        mask_func: callable sample mask class.
        img_size: image size.
        img_mode: 'B1000' or 'ADC'.
        net_mode: 'unet', 'wnet', 'varnet' or 'hybridvarnet'
        '''
        self.data_array = np.load(path)
        self.mask_func = mask_func
        self.img_size = img_size
        self.img_mode = img_mode
        self.net_mode = net_mode

    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, islice):
        mask = self.mask_func()[0]  # [H,W]
        img_slice = self.data_array[islice]
        if self.img_mode == 'ADC':
            img_slice = img_slice[3+256*256:].reshape(256, 256)
        else:  # B1000
            img_slice = img_slice[3:3+256*256].reshape(256, 256)
        mask = torch.from_numpy(mask).float()  # [H,W]
        img_slice = torch.from_numpy(img_slice).float()
        if self.img_size != 256:
            mask = T.Resize(size=self.img_size)(mask[None, ...]).squeeze(0)
            img_slice = T.Resize(size=self.img_size)(img_slice[None, ...]).squeeze(0)

        mask = mask[..., None]  # [H,W,1]
        img_slice = img_slice[..., None]  # [H,W,1]
        mask = torch.cat((mask, mask), dim=-1)  # [H,W,2]
        img_slice = torch.cat((img_slice, torch.zeros_like(img_slice)), dim=-1)  # [H,W,2]

        # ====== k-space undersampling ======
        kspace = fastmri.fft2c(img_slice)  # [H,W,2]
        kspace_masked = kspace * mask
        img_slice_masked = fastmri.ifft2c(kspace_masked)  # [H,W,2]

        img_full = fastmri.complex_abs(img_slice)  # [H,W]
        img_masked = fastmri.complex_abs(img_slice_masked)
        kspace = kspace[None, ...]  # [1(Nc),H,W,2]
        kspace_masked = kspace_masked[None, ...]

        if self.net_mode == 'unet':
            mask = mask[..., 0]  # [H,W]
            mask = mask[None, ...]  # [1,H,W]
            img_full = img_full[None, ...]  # [1,H,W]
            img_masked = img_masked[None, ...]

            # img [B,1,H,W], mask [B,1,H,W]
            return img_masked, img_full, mask

        else:  # wnet, varnet or hybridvarnet
            mask = mask[None, ...]  # [1,H,W,2]

            # kspace [B,1,H,W,2], mask [B,1,H,W,2], img [B,H,W]
            return kspace_masked, kspace, mask, img_full


class DatasetB1000Diffusion(Dataset):
    def __init__(
            self,
            path,
            mask_func,
            img_size=256,
            img_mode='B1000',
    ):
        '''
        path: path of the dataset.
        mask_func: callable sample mask class.
        img_size: image size.
        img_mode: 'B1000' or 'ADC'.
        net_mode: 'unet', 'wnet', 'varnet' or 'hybridvarnet'
        '''
        self.data_array = np.load(path)
        self.mask_func = mask_func
        self.img_size = img_size
        self.img_mode = img_mode

    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, islice):
        mask, mask_fold = self.mask_func()  # [B,H,W], [B,H,W]
        mask = mask[0]
        img_slice = self.data_array[islice]
        if self.img_mode == 'ADC':
            img_slice = img_slice[3+256*256:].reshape(256, 256)
        else:  # B1000
            img_slice = img_slice[3:3+256*256].reshape(256, 256)
        mask = torch.from_numpy(mask).float()  # [H,W]
        img_slice = torch.from_numpy(img_slice).float()
        if self.img_size != 256:
            mask = T.Resize(size=self.img_size)(mask[None, ...]).squeeze(0)
            img_slice = T.Resize(size=self.img_size)(img_slice[None, ...]).squeeze(0)

        mask = mask[..., None]  # [H,W,1]
        img_slice = img_slice[..., None]  # [H,W,1]
        mask = torch.cat((mask, mask), dim=-1)  # [H,W,2]
        img_slice = torch.cat((img_slice, torch.zeros_like(img_slice)), dim=-1)  # [H,W,2]

        # ====== k-space undersampling ======
        kspace = fastmri.fft2c(img_slice)  # [H,W,2]
        kspace_masked = kspace * mask
        img_slice_masked = fastmri.ifft2c(kspace_masked)  # [H,W,2]

        img_full = fastmri.complex_abs(img_slice)  # [H,W]
        img_masked = fastmri.complex_abs(img_slice_masked)
        kspace = kspace[None, ...]  # [1(Nc),H,W,2]
        kspace_masked = kspace_masked[None, ...]

        # [B,Nc,H,W,2], [B,1,H,W], [B,1,H/patch_size,W/patch_size]
        return kspace, mask[..., 0].unsqueeze(0), mask_fold
