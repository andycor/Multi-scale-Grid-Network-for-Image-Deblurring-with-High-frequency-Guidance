import torch
import torch.nn as nn
from config import cfg
from utils.network_utils import *
from skimage import measure
import random


#
# Deblurring Loss
#

def bceloss(input, target):
    n, c, h, w = input.size()

    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)

    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight[ignore_index] = 0

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
    return loss

def mseLoss(output, target):
    mse_loss = nn.MSELoss(reduction ='elementwise_mean')
    MSE = mse_loss(output, target)
    return MSE

def randomPatchLoss(output, target):
    crop_size = int(cfg.DATA.CROP_IMG_SIZE[0]/4)
    # print(output.size())
    output_size = int(output[0].size()[2])
    x_start = random.randint(0, output_size - crop_size-1)
    y_start = random.randint(0, output_size - crop_size-1)
    # print(x_start+crop_size,y_start+crop_size)
    return mseLoss(output[:,:,x_start+crop_size,y_start+crop_size],target[:,:,x_start+crop_size,y_start+crop_size])

def L1loss(output, target):
    L1_loss = nn.L1Loss()
    L1 = L1_loss(output,target)
    return L1

def SSIM(output, target, n_batch):
    # return 0
    output_arr = tensor2im(output)
    target_arr = tensor2im(target)
    ssim_sum = 0
    for i in range(output_arr.shape[0]):
        ssim_sum += measure.compare_ssim(output_arr[i], target_arr[i], multichannel=True)
    return ssim_sum/n_batch

def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)


def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

def featureLoss(F_encoder, F_resencoder,img, clearnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    features_real = clearnet(img)
    features_real.detach()
    # features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')
    # print(type(F_encoder))
    # print(type(F_resencoder))
    # print(type(features_real))

    loss = mse_loss(F_encoder+F_resencoder, features_real)

    return loss, features_real

# def gradeLoss(color, gray, delta=0.0006):
#     # print(color)
#     # print(gray)
#     n,c,h,w = color.size()
#     x1 = color[:, :, :, :w-1]
#     x2 = color[:, :, :, 1:]
#     Ix = F.pad(x1-x2, [0, 1, 0, 0, 0, 0], "constant", 0)
#     y1 = color[:, :, :h - 1, :]
#     y2 = color[:, :, 1:, :]
#     Iy = F.pad(y1-y2, [0, 0, 0, 1, 0, 0], 'constant', 0)
#     I = torch.abs(torch.cat([Ix, Iy], 2))
#     x1 = gray[:, :, :, :w - 1]
#     x2 = gray[:, :, :, 1:]
#     Gx = F.pad(x1-x2, [0, 1, 0, 0, 0, 0], "constant", 0)
#     y1 = gray[:, :, :h - 1, :]
#     y2 = gray[:, :, 1:, :]
#     Gy = F.pad(y1-y2, [0, 0, 0, 1, 0, 0], 'constant', 0)
#     G = torch.abs(torch.cat([Gx, Gy], 2))
#     # print(I.size())
#     # print(G.size())
#     # numerator = 2 * G.mul(I + delta)
#     # denominator = G.pow(2) + (I + delta).pow(2)
#     # # print(numerator.size())
#     # # print(denominator.size())
#     # loss = torch.tensor(1).cuda() - torch.mean(torch.div(numerator, denominator))
#
#     mse_loss = nn.MSELoss(reduction='elementwise_mean')
#     loss = mse_loss(I, G)
#
#     return loss


def gradeLoss(color, gray, delta=0.0006):
    # print(color)
    # print(gray)
    n, c, h, w = color.size()
    x1 = color[:, :, :, :w - 1]
    x2 = color[:, :, :, 1:]
    Ix = F.pad(x1 - x2, [0, 1, 0, 0, 0, 0], "constant", 0)
    y1 = color[:, :, :h - 1, :]
    y2 = color[:, :, 1:, :]
    Iy = F.pad(y1 - y2, [0, 0, 0, 1, 0, 0], 'constant', 0)
    I = torch.abs(Ix) + torch.abs(Iy)
    x1 = gray[:, :, :, :w - 1]
    x2 = gray[:, :, :, 1:]
    Gx = F.pad(x1 - x2, [0, 1, 0, 0, 0, 0], "constant", 0)
    y1 = gray[:, :, :h - 1, :]
    y2 = gray[:, :, 1:, :]
    Gy = F.pad(y1 - y2, [0, 0, 0, 1, 0, 0], 'constant', 0)
    G = torch.abs(Gx) + torch.abs(Gy)
    pooling = nn.MaxPool2d(kernel_size=20,stride=1,padding=((20-1)//2))
    I = pooling(I)
    G = pooling(G)
    # print(I.size())
    # print(G.size())

    # numerator = 2 * G.mul(I + delta)
    # denominator = G.pow(2) + (I + delta).pow(2)
    # # print(numerator.size())
    # # print(denominator.size())
    # loss = torch.tensor(1).cuda() - torch.mean(torch.div(numerator, denominator))

    mse_loss = nn.MSELoss(reduction='elementwise_mean')
    loss = mse_loss(I, G)
    # print(loss)

    return loss

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().detach().numpy()
    #print(image_numpy.shape)
    image_numpy = np.transpose(image_numpy,(0,2,3,1))
    #print(image_numpy.shape)
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    #print(image_numpy.shape)
    return image_numpy.astype(imtype)

