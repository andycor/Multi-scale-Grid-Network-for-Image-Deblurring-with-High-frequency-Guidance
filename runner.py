#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

import matplotlib
import os
import sys

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.build import bulid_net
import torch

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [cuda]', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--phase', dest='phase', help='phase of CNN', default=cfg.NETWORK.PHASE, type=str)
    parser.add_argument('--net', dest='net', help='net of deblurnet', default=cfg.NETWORK.NET, type=str)
    parser.add_argument('--name', dest='name', help='name of the task', default=cfg.NETWORK.NAME, type=str)
    parser.add_argument('--edge_loss', dest='edge_loss', help='edge_loss', default=cfg.NETWORK.EDGE_LOSS, type=str)
    parser.add_argument('--module', dest='module', help='module of net work', default=cfg.NETWORK.MODULE, type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=cfg.CONST.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--crop_size', dest='crop_size', help='crop size', default=256, type=int)
    parser.add_argument('--channel', dest='channel', help='channel size', default=cfg.NETWORK.CHANNEL, type=int)
    parser.add_argument('--num_resblock', dest='num_resblock', help='number of resblock', default=cfg.NETWORK.NUM_RESBLOCK, type=int)
    parser.add_argument('--data_augmentation', dest='data_augmentation', help='use data_augmentation or not', default=cfg.NETWORK.DATA_AUGMENTATION, type=int)
    parser.add_argument('--use_pretrained_model', dest='use_pretrained_model', help='use pretrained model or not', default=cfg.TRAIN.USE_PRETRAINED_MODEL, type=int)

    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=cfg.CONST.WEIGHTS, type=str)
    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default=cfg.DIR.DATASET_ROOT, type=str)
    parser.add_argument('--edge_dir', dest='edge_dir', help='Set dataset root_path', default='shape_edge', type=str)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args

def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.phase is not None:
        cfg.NETWORK.PHASE = args.phase
    if args.net is not None:
        cfg.NETWORK.NET = args.net
    if args.name is not None:
        cfg.NETWORK.NAME = args.name
    if args.edge_loss is not None:
        cfg.NETWORK.EDGE_LOSS = args.edge_loss
    if args.module is not None:
        cfg.NETWORK.MODULE = args.module
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    if args.data_path is not None:
        cfg.DIR.DATASET_ROOT = args.data_path
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.channel is not None:
        cfg.NETWORK.CHANNEL = args.channel
    if args.crop_size is not None:
        cfg.DATA.CROP_IMG_SIZE = [args.crop_size, args.crop_size]
    if args.batch_size is not None:
        cfg.CONST.TRAIN_BATCH_SIZE = args.batch_size
        if cfg.CONST.TRAIN_BATCH_SIZE > 1:
            cfg.TRAIN.NUM_EPOCHES = int(cfg.TRAIN.NUM_EPOCHES * cfg.CONST.TRAIN_BATCH_SIZE * 0.6)
            id = 0
            for i in cfg.TRAIN.DEBLURNET_LR_MILESTONES:
                cfg.TRAIN.DEBLURNET_LR_MILESTONES[id] = int(i * cfg.CONST.TRAIN_BATCH_SIZE * 0.6)
                id = id+1
            # cfg.TRAIN.DEBLURNET_LR_MILESTONES = int(cfg.TRAIN.DEBLURNET_LR_MILESTONES * cfg.CONST.TRAIN_BATCH_SIZE * 0.7)

        #cfg.CONST.TEST_BATCH_SIZE = args.batch_size
        if cfg.NETWORK.PHASE == 'test':
            cfg.CONST.TEST_BATCH_SIZE = 1
    if args.num_resblock is not None:
        cfg.NETWORK.NUM_RESBLOCK = args.num_resblock
    if args.data_augmentation is not None:
        cfg.NETWORK.DATA_AUGMENTATION = args.data_augmentation
    if args.use_pretrained_model is not None:
        cfg.TRAIN.USE_PRETRAINED_MODEL = args.use_pretrained_model
    if args.edge_dir is not None:
        pass


    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    print('CUDA DEVICES NUMBER: ' + str(torch.cuda.device_count()))

    # Setup Network & Start train/test process
    bulid_net(cfg)


if __name__ == '__main__':
    main()
