import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from utils.evaluation_utils import *



def train_unet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the U-Net.
    :param train_dataloader: training dataloader.
    :param test_dataloader: test dataloader.
    :param optimizer: optimizer.
    :param loss: loss function object.
    :param net: network object.
    :param device: device, gpu or cpu.
    :param NUM_EPOCH: number of epoch, default=5.
    :param show_step: int, default=-1. Steps to show intermediate loss during training. -1 for not showing.
    :param show_test: flag. Whether to show test after training.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)
            if i==0 and idx==0:
                print('X.shape:', X.shape)
                print('mask.shape:', mask.shape)
                print('y_pred.shape:', y_pred.shape)

            optimizer.zero_grad()
            loss_train = loss(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

        if i == 0 or (i + 1) % 5 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL+'model_ck'+str(i + 1)+'.pt')

    # test model
    if show_test:
        nmse, psnr, ssim = test_unet(
            test_dataloader,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL+'model_ck.pt')
    print('MODEL SAVED.')

    return net


def test_unet(
        test_dataloader,
        net,
        device):
    '''
    Test the reconstruction performance. U-Net.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            X, y, mask = data

            X = X.to(device).float()  #[B,1,H,W]
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach()  # [B,1,H,W]
            pred = y_pred.detach()
            tg = tg.squeeze(1)  # [B,H,W]
            pred = pred.squeeze(1)
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            if idx==0:
                print('tg.shape:', tg.shape)
                print('pred.shape:', pred.shape)

            nmseb = 0
            psnrb = 0
            ssimb = 0
            B = tg.shape[0]
            for idxs in range(B):
                nmseb += calc_nmse_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
                psnrb += calc_psnr_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
                ssimb += calc_ssim_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
            nmseb /= B
            psnrb /= B
            ssimb /= B
            nmse += nmseb
            psnr += psnrb
            ssim += ssimb

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def train_wnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_mid,
        loss_img,
        alpha,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the WNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = alpha * loss_mid(k_pred_mid, yk) + loss_img(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

        if i == 0 or (i + 1) % 5 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL+'model_ck'+str(i + 1)+'.pt')

    # test model
    if show_test:
        nmse, psnr, ssim = test_wnet(
            test_dataloader,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL+'model_ck.pt')
    print('MODEL SAVED.')

    return net


def test_wnet(
        test_dataloader,
        net,
        device):
    '''
    Test the reconstruction performance. WNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def train_varnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the VarNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = loss(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

        if i == 0 or (i + 1) % 5 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL+'model_ck'+str(i + 1)+'.pt')

    # test model
    if show_test:
        nmse, psnr, ssim = test_varnet(
            test_dataloader,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL+'model_ck.pt')
    print('MODEL SAVED.')

    return net


def test_varnet(
        test_dataloader,
        net,
        device):
    '''
    Test the reconstruction performance. VarNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
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

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim
