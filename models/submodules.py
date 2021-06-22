#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

import torch.nn as nn
import torch
import numpy as np
from config import cfg

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias, groups=groups),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)
    )

def predict_disp(in_channels):
    return nn.Conv2d(in_channels,1,kernel_size=3,stride=1,padding=1,bias=True)

def predict_disp_bi(in_channels):
    return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=True)

def up_disp_bi():
    return nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

def predict_occ(in_channels):
    return nn.Conv2d(in_channels,1,kernel_size=3,stride=1,padding=1,bias=True)

def predict_occ_bi(in_channels):
    return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=True)

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

def res_blocks(in_channels, kernel_size=3, dilation=[1,1], bias=True, num=9):
    model = []
    for i in range(num):
        model += [ResnetBlock(in_channels, kernel_size, dilation, bias)]
    res = nn.Sequential(*model)
    return res
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


def gatenet(bias=True):
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        resnet_block(16, kernel_size=1),
        nn.Conv2d(16, 1, kernel_size=1, padding=0),
        nn.Sigmoid()
    )

def depth_sense(in_channels, out_channels, kernel_size=3, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,padding=((kernel_size - 1) // 2)*dilation, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True),
        resnet_block(out_channels, kernel_size= 3),
    )

def conv2x(in_channels, kernel_size=3,dilation=[1,1], bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)
    )


def ms_dilate_block(in_channels, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels, kernel_size, dilation, bias)

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Conv2d(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output

class Self_Channel_Attention(nn.Module):
    """ Self channel attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Channel_Attention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        # print("m_batchsize:"+str(m_batchsize))
        # print("C:"+str(C))
        # print("width:"+str(width))
        # print("height:"+str(height))

        proj_query_channel = self.query_conv_channel(x).view(m_batchsize, -1, width * height).permute(0, 2,
                                                                                                      1)  # B X CX(N)
        proj_key_channel = self.key_conv_channel(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_key_channel, proj_query_channel)  # transpose check bmm:batch matrix multiply
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(m_batchsize, C, width, height)

        # print(x.size())
        # print(proj_query_channel.size())
        # print(proj_key_channel.size())
        # print(energy.size())
        # print(out.size())

        out = self.gamma * out + x
        # return out, attention
        return out


class Self_Position_Attention(nn.Module):
    """ Self Position attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Position_Attention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        in_dim2 = max(in_dim // 8, 1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        y = x
        # print("x.size(): "+str(x.size()))
        # print("m_batchsize:"+str(m_batchsize))
        # print("C:"+str(C))
        # print("width:"+str(width))
        # print("height:"+str(height))
        n_d = 1
        if width >= 128:
            n_d = width // 128

        if n_d >= 2:
            pooling = nn.AvgPool2d((n_d, n_d), stride=(n_d, n_d))
            y = pooling(y)

        m_batchsize, C, width, height = y.size()

        # pooling = nn.AvgPool2d((n_downsampling, n_downsampling), stride=(n_downsampling, n_downsampling))
        # x = pooling(x)
        # print("pooling: x.size(): " + str(x.size()))
        # upsample = nn.Upsample(scale_factor=n_downsampling, mode='bilinear', align_corners=True)
        # x = upsample(x)
        # print("upsample: x.size(): " + str(x.size()))

        proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check bmm:batch matrix multiply
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # print(y.size())
        # print(proj_query.size())
        # print(proj_key.size())
        # print(energy.size())
        # print(out.size())

        if n_d >= 2:
            upsample = nn.Upsample(scale_factor=n_d, mode='bilinear', align_corners=True)
            out = upsample(out)

        out = self.gamma * out + x

        # return out, attention
        return out

