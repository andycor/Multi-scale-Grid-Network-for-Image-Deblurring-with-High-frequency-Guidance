#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import torchvision

from losses.multiscaleloss import *
from time import time

from core.test_deblur import test_deblurnet
from models.VGG19 import VGG19


def train_deblurnet(cfg, init_epoch, train_data_loader, val_data_loader, deblurnet, deblurnet_solver,
                    deblurnet_lr_scheduler, ckpt_dir, train_writer, val_writer, Best_Img_PSNR, Best_Epoch):
    # Training loop
    print("Andy: Processing train_deblurnet\n")
    Best_Img_SSIM = 0
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        test_time = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        mse_losses = utils.network_utils.AverageMeter()
        mse_edge_losses = utils.network_utils.AverageMeter()
        percept_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()
        img_SSIMs = utils.network_utils.AverageMeter()

        # # Adjust learning rate
        # deblurnet_lr_scheduler.step()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        vggnet = VGG19()
        if torch.cuda.is_available():
            vggnet = torch.nn.DataParallel(vggnet).cuda()

        for batch_idx, (_, images,images2) in enumerate(train_data_loader):
            # Measure data time

            data_time.update(time() - batch_end_time)
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            imgs2 = [utils.network_utils.var_or_cuda(img) for img in images2]

            # switch models to training mode
            deblurnet.train()
            if cfg.NETWORK.MODULE == 'Multitask':
                img_blur, img_shape = imgs
                img_edge = imgs2[0]
                output_img_shape, output_img_edge = deblurnet(img_blur)
            elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                img_blur, img_shape = imgs
                img_edge = imgs2[0]
                output_img_shapes, output_img_edges = deblurnet(img_blur)
            else:
                img_blur, img_shape = imgs
                output_img_shape = deblurnet(img_blur)

            if cfg.NETWORK.MODULE == 'MultiScale':
                img_shapes = multiScaleImage(img_shape)
                mse_loss = 0
                percept_loss = 0
                bl = [1, 1, 1]
                Size = len(output_img_shapes)
                for i in range(Size):
                    mse_loss += mseLoss(output_img_shapes[i], img_shapes[i%3])
                    percept_loss += perceptualLoss(output_img_shapes[i], img_shapes[i%3], vggnet)
                deblur_loss = (mse_loss + 0.01 * percept_loss)/3
            elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                img_shapes = multiScaleImage(img_shape)
                img_edges = multiScaleImage(img_edge)
                mse_loss = 0
                percept_loss = 0
                for i in range(3):
                    mse_loss += mseLoss(output_img_shapes[i], img_shapes[i])
                    percept_loss += perceptualLoss(output_img_shapes[i], img_shapes[i], vggnet)
                deblur_loss = (mse_loss + 0.01 * percept_loss) / 3
                mse_edge_loss = 0
                for i in range(3):
                    mse_edge_loss += mseLoss(output_img_edges[i], img_edges[i])
                deblur_loss += 0.01 * mse_edge_loss/3
            elif cfg.NETWORK.MODULE == 'Multitask':
                mse_loss = mseLoss(output_img_shape, img_shape)
                percept_loss = perceptualLoss(output_img_shape, img_shape, vggnet)
                deblur_loss = mse_loss + 0.01 * percept_loss

                mse_edge_loss = mseLoss(output_img_edge, img_edge)
                deblur_loss += 0.1 * mse_edge_loss
            else:
                mse_loss = mseLoss(output_img_shape, img_shape)
                percept_loss = perceptualLoss(output_img_shape, img_shape, vggnet)
                deblur_loss = mse_loss + 0.01 * percept_loss


            if cfg.NETWORK.MODULE == 'MultiScale':
                img_PSNR = 0
                img_SSIM = 0

                img_PSNR += PSNR(output_img_shapes[0], img_shapes[0])
                img_SSIM += SSIM(output_img_shapes[0], img_shapes[0], cfg.CONST.TRAIN_BATCH_SIZE)
            elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                img_PSNR = PSNR(output_img_shapes[0], img_shape)
                img_SSIM = SSIM(output_img_shapes[0], img_shape, cfg.CONST.TRAIN_BATCH_SIZE)

            else:
                img_PSNR = PSNR(output_img_shape, img_shape)
                img_SSIM = SSIM(output_img_shape, img_shape, cfg.CONST.TRAIN_BATCH_SIZE)

            # Gradient decent
            deblurnet_solver.zero_grad()
            deblur_loss.backward()
            deblurnet_solver.step()

            # update_avg_losses
            mse_losses.update(mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            percept_losses.update(percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            img_SSIMs.update(img_SSIM, cfg.CONST.TRAIN_BATCH_SIZE)
            if cfg.NETWORK.MODULE == 'Multitask' or cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                mse_edge_losses.update(mse_edge_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('MSELoss__TRAIN', mse_loss.item(), n_itr)
            train_writer.add_scalar('PerceptLoss__TRAIN', percept_loss.item(), n_itr)
            if cfg.NETWORK.MODULE == 'Multitask' or cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                train_writer.add_scalar('EdgeLoss__TRAIN', mse_edge_loss.item(), n_itr)
            train_writer.add_scalar('DeblurLoss__TRAIN', deblur_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                if cfg.NETWORK.MODULE == 'Multitask' or cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t Loss {4} [{5}, {6},{7}]\t PSNR {8} SSIM {9}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches,
                                  deblur_losses, mse_losses, percept_losses, mse_edge_losses, img_PSNRs, img_SSIMs))
                else:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t DeblurLoss {4} [{5}, {6}] \t PSNR {7} SSIM {8}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches,
                                  deblur_losses, mse_losses, percept_losses, img_PSNRs, img_SSIMs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:
                img_blur = images[0][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_shape = images[1][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_edge = images2[0][0][0, :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                if cfg.NETWORK.MODULE == 'MultiScale':
                    out_imgs = []
                    out_imgs.append(output_img_shapes[0][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    out_imgs.append(output_img_shapes[1][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    #out_imgs.append(output_img_shapes[2][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    result = torch.cat([img_blur, img_shape, out_imgs[0]], 1)
                elif cfg.NETWORK.MODULE == 'MultiScale_with_edge':
                    out_imgs = []
                    out_imgs.append(output_img_shapes[0][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    out_imgs.append(output_img_shapes[1][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    #out_imgs.append(output_img_shapes[2][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1))
                    out_edge = output_img_edges[0][0][0, :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    result = torch.cat([img_blur, img_shape, out_imgs[0], out_edge], 1)
                elif cfg.NETWORK.MODULE == 'Multitask':
                    out_img = output_img_shape[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    out_edge = output_img_edge[0][0, :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    result = torch.cat([img_blur, img_shape, img_edge, out_img, out_edge], 1)
                else:
                    out_img = output_img_shape[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    result = torch.cat([img_blur,img_shape,out_img], 1)

                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)
        train_writer.add_scalar('EpochSSIM_0_TRAIN', img_SSIMs.avg, epoch_idx + 1)

        # Adjust learning rate
        deblurnet_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t DeblurLoss_avg {3}\t ImgPSNR_avg {4}\t ImgSSIM_avg {5}'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, deblur_losses.avg,
                      img_PSNRs.avg, img_SSIMs.avg))
        print('Best PSNR: {0}\t  Best SSIM: {1} Best Epoch: {2}\n'.format(Best_Img_PSNR, Best_Img_SSIM,  Best_Epoch))
        print("learning rate : " + str(deblurnet_lr_scheduler.get_lr()[0]))

        # Validate the training models
        img_PSNR = 0
        img_SSIM = 0
        if epoch_idx % 10 == 0:
            img_PSNR,img_SSIM = test_deblurnet(cfg, epoch_idx, val_data_loader, deblurnet, val_writer)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_deblur_checkpoints(
                os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                Best_Epoch)
        if img_PSNR > Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNR
            Best_Img_SSIM = img_SSIM
            Best_Epoch = epoch_idx + 1
            utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'),
                                                        epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                                                        Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


