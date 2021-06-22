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

def test_deblurnet(cfg, epoch_idx, test_data_loader, deblurnet, test_writer):

    # Testing loop
    n_batches = len(test_data_loader)
    test_epe  = dict()
    # Batch average meterics
    batch_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs = utils.network_utils.AverageMeter()
    img_SSIMs = utils.network_utils.AverageMeter()
    #img_PSNRs.reset()
    #img_SSIMs.reset()
    batch_end_time = time()

    test_psnr = dict()
    g_names= 'init'
    save_num = 0
    for batch_idx, (names, images, images2) in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)
        if not g_names == names:
            g_names = names
            save_num = 0
        save_num = save_num+1
        # Switch models to testing mode
        deblurnet.eval()

        if cfg.NETWORK.PHASE == 'test':
            assert (len(names) == 1)
            name = names[0]
            if not name in test_psnr:
                test_psnr[name] = {
                    'n_samples': 0,
                    'psnr': []
                }

        with torch.no_grad():
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            imgs2 = [utils.network_utils.var_or_cuda(img) for img in images2]

            batch_end_time = time()
            # Test the decoder
            if cfg.NETWORK.MODULE == 'Multitask':
                img_blur, img_shape = imgs
                img_edge = imgs2[0]
                output_img_shape, output_img_edge = deblurnet(img_blur)
            elif cfg.NETWORK.MODULE == 'MultiScale':
                img_blur, img_shape = imgs
                output_img_shapes = deblurnet(img_blur)
            elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                img_blur, img_shape = imgs
                img_edge = imgs2
                output_img_shapes, output_img_edges = deblurnet(img_blur)
            else:
                img_blur, img_shape = imgs
                output_img_shape = deblurnet(img_blur)
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Append loss and accuracy to average metrics
            if cfg.NETWORK.MODULE == 'MultiScale':
                img_PSNR = PSNR(output_img_shapes[0], img_shape)
                img_SSIM = SSIM(output_img_shapes[0], img_shape, cfg.CONST.TEST_BATCH_SIZE)
            elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                img_PSNR = PSNR(output_img_shapes[0], img_shape)
                img_SSIM = SSIM(output_img_shapes[0], img_shape, cfg.CONST.TRAIN_BATCH_SIZE)
            else :
                img_PSNR = PSNR(output_img_shape, img_shape)
                img_SSIM = SSIM(output_img_shape, img_shape, cfg.CONST.TRAIN_BATCH_SIZE)

            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TEST_BATCH_SIZE)
            img_SSIMs.update(img_SSIM, cfg.CONST.TEST_BATCH_SIZE)

            if cfg.NETWORK.PHASE == 'test':
                test_psnr[name]['n_samples'] += 1
                test_psnr[name]['psnr'].append(img_PSNR)


            # Print result
            if (batch_idx+1) % cfg.TEST.PRINT_FREQ == 0:
                print('[TEST] [Epoch {0}/{1}][Batch {2}/{3}]\t ImgPSNR {4}\t ImgSSIM {5}\t batch_time {6}'
                      .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, img_PSNRs, img_SSIMs, batch_time))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:
                if cfg.NETWORK.MODULE == 'MultiScale' or cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                    output_img_shape = output_img_shapes[0]
                if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test', 'resume']:
                    test_writer.add_image('IMG_BLUR'+str(batch_idx+1),
                                          images[0][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)
                    test_writer.add_image('IMG_SHAPE'+str(batch_idx+1),
                                          images[1][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)

                test_writer.add_image('UT_IMG_SHAPE'+str(batch_idx+1), output_img_shape[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)

            if cfg.NETWORK.PHASE == 'test' and cfg.TEST.IS_WRITE == 1:
                out_dir = os.path.join(cfg.DIR.OUT_PATH,cfg.NETWORK.NET+'_'+cfg.NETWORK.NAME,'result_shape')
                if not os.path.isdir(out_dir):
                    mkdir(out_dir)
                print('saving : '+out_dir+'/'+names[0])
                if cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                    cv2.imwrite(out_dir + '/' + names[0],
                                (output_img_shapes[0].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                    np.uint8),
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                    # cv2.imwrite(out_dir + '/lr1' + names[0],
                    #             (output_img_shapes[1].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                    #                 np.uint8),
                    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
                    # cv2.imwrite(out_dir + '/lr2' + names[0],
                    #             (output_img_shapes[2].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                    #                 np.uint8),
                    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

                else:
                    cv2.imwrite(out_dir+'/'+names[0], (output_img_shape.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8),
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 5])


    if cfg.NETWORK.PHASE == 'test':

        # Output test results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg) + " SSIM:"+str(img_SSIMs.avg))
        for name in test_psnr:
            # test_psnr[name]['psnr'] = np.mean(test_psnr[name]['psnr'], axis=0)
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg) + " SSIM:" + str(img_SSIMs.avg))

        result_file = open(os.path.join(cfg.DIR.OUT_PATH,cfg.NETWORK.NET+'_'+cfg.NETWORK.NAME, 'test_result.txt'), 'w')
        sys.stdout = result_file
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg) + " SSIM:"+str(img_SSIMs.avg))
        for name in test_psnr:
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        result_file.close()
    else:
        # Output val results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg) + " SSIM:"+str(img_SSIMs.avg))
        print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t ImgPSNR_avg {3}\t ImgSSIM_avg {4}\n'
              .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, img_PSNRs.avg, img_SSIMs.avg))

        # Add testing results to TensorBoard
        test_writer.add_scalar('DeblurNet/EpochPSNR__TEST', img_PSNRs.avg, epoch_idx + 1)
        test_writer.add_scalar('DeblurNet/EpochSSIM__TEST', img_SSIMs.avg, epoch_idx + 1)

        return img_PSNRs.avg,img_SSIMs.avg