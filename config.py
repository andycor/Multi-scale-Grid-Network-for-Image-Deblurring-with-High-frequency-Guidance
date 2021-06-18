#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

from easydict import EasyDict as edict
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = '2'                   # '0' / 'all'
__C.CONST.NUM_WORKER                    = 4                      # number of data workers
__C.CONST.TRAIN_BATCH_SIZE              = 1
__C.CONST.TEST_BATCH_SIZE               = 1
__C.CONST.WEIGHTS                       = 'NONE'

#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'GoPro'          # predict, GoPro

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH = './data/output'


# For GoPro_Dataset
# cfg.DATASET.DATASET_NAME == 'GoPro':
__C.DIR.DATASET_ROOT                    = '../GoPro'
__C.DIR.Train_Blur_image_PATH           = __C.DIR.DATASET_ROOT + '/train/A'
__C.DIR.Train_Shape_image_PATH          = __C.DIR.DATASET_ROOT + '/train/B'
__C.DIR.Train_Edge_image_PATH           = __C.DIR.DATASET_ROOT + '/train/log_edge'
__C.DIR.Train_Deblured_image_PATH       = '/home/newdisk/andy/GoPro/GoPro_result/my_train_result/result_shape'


__C.DIR.Test_Blur_image_PATH            = __C.DIR.DATASET_ROOT + '/test/A'
__C.DIR.Test_Shape_image_PATH           = __C.DIR.DATASET_ROOT + '/test/B'
__C.DIR.Test_Edge_image_PATH            = __C.DIR.DATASET_ROOT + '/test/DCT_edge'
__C.DIR.Test_Deblured_image_PATH        = '/home/newdisk/andy/GoPro/GoPro_result/my_test_result/result_shape'
#

# __C.DIR.Predict_image_PATH              = '/home/newdisk/andy/DataSet/Lai'
__C.DIR.Predict_image_PATH              = __C.DIR.DATASET_ROOT + '/test/A'

#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.DIV_DISP                       = 40.0                    # 40.0 for disparity
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.NET                         = 'model'
__C.NETWORK.LEAKY_VALUE                 = 0.1
__C.NETWORK.BATCHNORM                   = False
__C.NETWORK.PHASE                       = 'test'
__C.NETWORK.MODULE                      = 'MultiScale_with_edge'
__C.NETWORK.NAME                        = 'final'
__C.NETWORK.EDGE_LOSS                   = 'mse'
__C.NETWORK.NUM_RESBLOCK                = 9
__C.NETWORK.DATA_AUGMENTATION           = 1                 # 0 or 1
__C.NETWORK.USE_DISCRIMINATOR           = 0
__C.NETWORK.USE_DEBLURED_DATASET        = 0
__C.NETWORK.CHANNEL                     = 40
#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PRETRAINED_MODEL          = 0
__C.TRAIN.PRETRAIN_DIR                  = './data/pretrained model/DAVANET/best-ckpt.pth.tar'
__C.TRAIN.USE_PERCET_LOSS               = True
__C.TRAIN.USE_GRADE_LOSS                = False
__C.TRAIN.NUM_EPOCHES                   = 502                      # maximum number of epoches
__C.TRAIN.BRIGHTNESS                    = .25
__C.TRAIN.CONTRAST                      = .25
__C.TRAIN.SATURATION                    = .25
__C.TRAIN.HUE                           = .25
__C.TRAIN.DEBLURNET_LEARNING_RATE       = 1e-4
__C.TRAIN.DEBLURNET_LR_MILESTONES       = [180,250,350,430,490]
# __C.TRAIN.DEBLURNET_LR_MILESTONES       = [30,50,70,430,490]      # use it for resume
__C.TRAIN.LEARNING_RATE_DECAY           = 0.3                     # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                     # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                     # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 300
__C.TRAIN.SAVE_FREQ                     = 100                     # weights will be overwritten every save_freq epoch

__C.LOSS                                = edict()


#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.IS_WRITE                       = 1                               #write the deblured image or not
__C.TEST.VISUALIZATION_NUM              = 3
__C.TEST.PRINT_FREQ                     = 100
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE           = 1
