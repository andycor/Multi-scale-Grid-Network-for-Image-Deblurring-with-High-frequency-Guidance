#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy
import os
import sys
import torch.backends.cudnn
import torch.utils.data
import numpy as np
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multiscaleloss import *
from time import time
import cv2

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def predict_deblurnet(cfg, epoch_idx, test_data_loader, deblurnet):

    # Testing loop
    n_batches = len(test_data_loader)
    test_epe  = dict()
    # Batch average meterics

    test_psnr = dict()
    g_names= 'init'
    save_num = 0
    time_avg = 0
    cnt = 0
    for batch_idx, (names, images, images2) in enumerate(test_data_loader):
        if not g_names == names:
            g_names = names
            save_num = 0
        save_num = save_num+1
        # Switch models to testing mode
        deblurnet.eval()

        assert (len(names) == 1)
        name = names[0]
        if not name in test_psnr:
            test_psnr[name] = {
                'n_samples': 0,
                'psnr': []
            }
        parm = {}
        for name2, parameters in deblurnet.named_parameters():
            print(name2, ':',parameters.abs().sum())
            parm[name2] = parameters.cpu().detach().numpy()
        with torch.no_grad():
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            img_blur, img_shape, img_edge = imgs
            batch_start_time = time()
            # Test the decoder
            if cfg.NETWORK.MODULE == 'Multitask':
                output_img_shape, output_img_edge = deblurnet(img_blur)
            elif cfg.NETWORK.MODULE == 'MultiScale':
                output_img_shapes, features  = deblurnet(img_blur)
            else:
                output_img_shape, features = deblurnet(img_blur)
            batch_end_time = time()
            cnt += 1

            # Append loss and accuracy to average metrics

            test_psnr[name]['n_samples'] += 1
            print('predict time: ' + str(batch_end_time-batch_start_time))
            time_avg += batch_end_time-batch_start_time



            out_dir = os.path.join(cfg.DIR.OUT_PATH,cfg.NETWORK.NET+'_'+cfg.NETWORK.NAME,'result_shape_predict')
            if not os.path.isdir(out_dir):
                mkdir(out_dir)
            print('saving : '+out_dir+'/'+names[0])
            if cfg.NETWORK.MODULE == 'MultiScale':
                cv2.imwrite(out_dir + '/' + names[0],
                            (output_img_shapes[0].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8),
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                # print(features.size())
                for i in range(features.size()[1]):
                    mmax = features[0][i].max()
                    cv2.imwrite(out_dir + '/feature/' + str(i) +names[0],
                                ((features/mmax).clamp(0.0, 1.0)[0][i].cpu().numpy() * 255.0).astype(
                                    np.uint8))
            else:
                cv2.imwrite(out_dir+'/'+names[0], (output_img_shape.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8),
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                # print(features.size())
                for i in range(features.size()[1]):
                    mmax = features[0][i].max()
                    cv2.imwrite(out_dir + '/feature/' + str(i) + names[0],
                                ((features / mmax).clamp(0.0, 1.0)[0][i].cpu().numpy() * 255.0).astype(
                                    np.uint8))

            if(cnt>5):break
    print('time_avg:' + str(time_avg/cnt))
    return