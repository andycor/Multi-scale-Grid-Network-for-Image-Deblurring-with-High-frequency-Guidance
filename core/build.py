#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

import os
import sys
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import models

from models.model import model





from datetime import datetime as dt
from tensorboardX import SummaryWriter

from core.train_deblur import train_deblurnet
from core.predict_deblur import predict_deblurnet
from core.test_deblur import test_deblurnet

from losses.multiscaleloss import *

def bulid_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    if cfg.NETWORK.DATA_AUGMENTATION == 1:
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
            utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
            #utils.data_transforms.RandomDownsample(),
            utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
            utils.data_transforms.RandomVerticalFlip(),
            utils.data_transforms.RandomHorizonFlip(),
            utils.data_transforms.RandomColorChannel(),
            utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
            utils.data_transforms.ToTensor(),
        ])
    else:
        train_transforms = utils.data_transforms.Compose([
            # utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
            utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
            utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
            # utils.data_transforms.RandomVerticalFlip(),
            # utils.data_transforms.RandomColorChannel(),
            # utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
            utils.data_transforms.ToTensor(),
        ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        #utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()
    if cfg.NETWORK.PHASE in ['train', 'resume', 'detection']:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

    test_data_loader   = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)


    # Set up networks
    deblurnet = models.__dict__[cfg.NETWORK.NET].__dict__[cfg.NETWORK.NET]()



    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.NET,
                                                utils.network_utils.count_parameters(deblurnet)))

    # Initialize weights of networks
    deblurnet.apply(utils.network_utils.init_weights_xavier)
    # Set up solver
    deblurnet_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, deblurnet.parameters()), lr=cfg.TRAIN.DEBLURNET_LEARNING_RATE,
                                        betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            deblurnet = torch.nn.DataParallel(deblurnet).cuda()
        else:
            deblurnet.cuda()

    # Summary writer for TensorBoard
    output_dir   = os.path.join(cfg.DIR.OUT_PATH, cfg.NETWORK.NET+'_'+cfg.NETWORK.NAME, '%s')
    log_dir      = output_dir % 'logs'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Img_PSNR    = 0
    if cfg.CONST.WEIGHTS == 'NONE':
        cfg.CONST.WEIGHTS = os.path.join(ckpt_dir, 'best-ckpt.pth.tar')
    if cfg.NETWORK.PHASE in ['test', 'resume', 'predict']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
        init_epoch = checkpoint['epoch_idx']+1
        Best_Img_PSNR = checkpoint['Best_Img_PSNR']
        Best_Epoch = checkpoint['Best_Epoch']
        print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
              .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))
        if cfg.NETWORK.PHASE =='test' or 'resume':
            init_epoch = 0
    if cfg.NETWORK.PHASE == 'train' and cfg.TRAIN.USE_PRETRAINED_MODEL == 1:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.TRAIN.PRETRAIN_DIR))
        pretrain_model = torch.load(cfg.TRAIN.PRETRAIN_DIR)
        model_dict = deblurnet.state_dict()
        pretrain_dict = pretrain_model['deblurnet_state_dict']
        # deblurnet.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_dict.items()})
        if torch.cuda.device_count() > 1:
            deblurnet.load_state_dict({'module.'+k: v for k, v in pretrain_dict.items()}, strict=False)
        else:
            deblurnet.load_state_dict(pretrain_dict, strict=False)


    # Set up learning rate scheduler to decay learning rates dynamically
    deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                  milestones=cfg.TRAIN.DEBLURNET_LR_MILESTONES,
                                                                  gamma=cfg.TRAIN.LEARNING_RATE_DECAY)


    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # train and val
        if cfg.NETWORK.USE_DISCRIMINATOR == 1:
            pass
        else :
            train_deblurnet(cfg, init_epoch, train_data_loader, test_data_loader, deblurnet, deblurnet_solver,
                        deblurnet_lr_scheduler, ckpt_dir, train_writer, test_writer, Best_Img_PSNR, Best_Epoch)
        return

    elif cfg.NETWORK.PHASE == 'test':
        assert os.path.exists(cfg.CONST.WEIGHTS), '[FATAL] Please specify the file path of checkpoint!'
        test_deblurnet(cfg, init_epoch, test_data_loader, deblurnet, test_writer)
        return
    elif cfg.NETWORK.PHASE == 'predict':
        assert os.path.exists(cfg.CONST.WEIGHTS), '[FATAL] Please specify the file path of checkpoint!'
        predict_deblurnet(cfg, init_epoch, test_data_loader, deblurnet)
        return

