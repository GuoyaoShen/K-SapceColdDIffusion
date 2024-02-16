import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from net.wnet.unet import NormUnet, DualCNormUnet



class WNet(nn.Module):
    """
    W-Net model.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        num_coil: int = 16,
        drop_prob: float = 0.0,
        use_attention: bool = False,
        use_res: bool = False,
    ):
        super().__init__()

        self.knet = DualCNormUnet(
            in_chans=in_chans*num_coil,
            out_chans=in_chans*num_coil,
            chans=chans,
            num_pools=num_pool_layers,
            drop_prob=drop_prob,
            use_attention=use_attention,
            use_res=use_res,
        )

        self.imgnet = NormUnet(
            in_chans=out_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            use_attention=use_attention,
            use_res=use_res,
        )

        self.dc = DC_block()

    def forward(self, xk: torch.Tensor, mask: torch.Tensor):

        predk = self.knet(xk)  # [B,Nc,H,W,2]
        predk = self.dc(predk, xk, mask)

        # k-space to img-space
        img_in = fastmri.ifft2c(predk)  # [B,Nc,H,W,2]
        img_in = fastmri.complex_abs(img_in)  # [B,Nc,H,W]
        img_in = fastmri.rss(img_in, dim=1)  # [B,H,W]
        img_in = img_in[:, None, :, :]  # [B,1,H,W]

        img_out = self.imgnet(img_in)
        img_out = img_out.squeeze(1)  # [B,H,W]

        return img_out, predk


class DC_block(nn.Module):
    '''
    Data consistency (DC) block.
    '''
    def __init__(self):
        super(DC_block, self).__init__()

    def forward(self, k_pred, k_sampled, mask):
        '''
        k_pred, k_sampled
        mask: binary mask
        '''
        mask_inv = torch.where(mask != 0, 0., 1.)
        k_dc = k_sampled + k_pred * mask_inv
        return k_dc