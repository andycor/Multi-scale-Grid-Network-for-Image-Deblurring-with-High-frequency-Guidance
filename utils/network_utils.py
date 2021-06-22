#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by andy

import os
import sys
import torch
import numpy as np
from datetime import datetime as dt
from config import cfg
import torch.nn.functional as F

import cv2

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_deblur_checkpoints(file_path, epoch_idx, deblurnet, deblurnet_solver, Best_Img_PSNR, Best_Epoch):
    print('[INFO] %s Saving checkpoint to %s ...\n' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'Best_Img_PSNR': Best_Img_PSNR,
        'Best_Epoch': Best_Epoch,
        'deblurnet_state_dict': deblurnet.state_dict(),
        'deblurnet_solver_state_dict': deblurnet_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)

def save_checkpoints(file_path, epoch_idx, dispnet, dispnet_solver, deblurnet, deblurnet_solver, Disp_EPE, Best_Img_PSNR, Best_Epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'Disp_EPE': Disp_EPE,
        'Best_Img_PSNR': Best_Img_PSNR,
        'Best_Epoch': Best_Epoch,
        'dispnet_state_dict': dispnet.state_dict(),
        'dispnet_solver_state_dict': dispnet_solver.state_dict(),
        'deblurnet_state_dict': deblurnet.state_dict(),
        'deblurnet_solver_state_dict': deblurnet_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_weight_parameters(model):
    return [param for name, param in model.named_parameters() if ('weight' in name)]

def get_bias_parameters(model):
    return [param for name, param in model.named_parameters() if ('bias' in name)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        # print("reset:\n")
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)

'''input Tensor: 2 H W'''
def graybi2rgb(graybi):
    assert(isinstance(graybi, torch.Tensor))
    global args
    _, H, W = graybi.shape
    rgb_1 = torch.zeros((3,H,W))
    rgb_2 = torch.zeros((3,H,W))
    normalized_gray_map = graybi / (graybi.max())
    rgb_1[0] = normalized_gray_map[0]
    rgb_1[1] = normalized_gray_map[0]
    rgb_1[2] = normalized_gray_map[0]

    rgb_2[0] = normalized_gray_map[1]
    rgb_2[1] = normalized_gray_map[1]
    rgb_2[2] = normalized_gray_map[1]
    return rgb_1.clamp(0,1), rgb_2.clamp(0,1)

def multiScaleImage(img):
    ans = []
    ans.append(img)
    downsample1 = F.interpolate(img, scale_factor=0.5)
    downsample2 = F.interpolate(img, scale_factor=0.25)
    ans.append(downsample1)
    ans.append(downsample2)
    return ans







