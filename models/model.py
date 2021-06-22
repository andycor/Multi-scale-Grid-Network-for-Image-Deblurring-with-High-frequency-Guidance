#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy

# from models.submodules import *
from torchsummary import summary
from thop import profile
from torchstat import stat
from models.submodules import *
import os
from torch.nn import functional as F
class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        ks = 3
        C = cfg.NETWORK.CHANNEL
        # encoder
        # self.encoder = Encoder(C)
        # self.decoder = Decoder(C)
        self.conv0 = conv(3, C, kernel_size=ks, stride=1)
        self.conv1_2 = res_blocks(C, kernel_size=ks, num=1)
        self.conv2_2 = res_blocks(2*C, kernel_size=ks, num=1)
        self.conv3_2 = res_blocks(4*C, kernel_size=ks, num=1)

        self.down1_1 = nn.Sequential(
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 2*C, kernel_size=ks, stride=2),
        )
        self.down2_1 = nn.Sequential(
            res_blocks(2*C, kernel_size=ks, num=3),
            conv(2*C, 4*C, kernel_size=ks, stride=2),
        )

        self.up1_1 = nn.Sequential(
            res_blocks(2*C, kernel_size=ks, num=3),
            upconv(2*C, C),
        )
        self.up2_1 = nn.Sequential(
            res_blocks(4*C, kernel_size=ks, num=3),
            upconv(4*C, 2*C),
        )

        self.aggD2_1 = nn.Sequential(
            conv(4*C, 2*C, kernel_size=ks, stride=1),
            res_blocks(2*C, kernel_size=ks, num=2),
        )
        self.aggD1_1 = nn.Sequential(
            conv(2*C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=2),
        )

        self.img_prd = nn.Sequential(
            conv(C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=1),
            conv(C, 3, kernel_size=ks, stride=1),
        )
        self.img_prd1 = nn.Sequential(
            conv(2*C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=1),
            conv(C, 3, kernel_size=ks, stride=1),
        )
        self.img_prd2 = nn.Sequential(
            conv(4*C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=1),
            conv(C, 3, kernel_size=ks, stride=1),
        )


    def forward(self, x):
        # encoder
        E1_1 = self.conv0(x)
        E2_1 = self.down1_1(E1_1)
        E3_1 = self.down2_1(E2_1)
        # decoder
        D3_1 = self.conv3_2(E3_1)
        D2_1 = self.aggD2_1(torch.cat([self.up2_1(D3_1), self.conv2_2(E2_1)], 1))
        D1_1 = self.aggD1_1(torch.cat([self.up1_1(D2_1), self.conv1_2(E1_1)], 1))

        img_prd = self.img_prd(D1_1)
        img_prd1 = self.img_prd1(D2_1)
        img_prd2 = self.img_prd2(D3_1)

        return D1_1, img_prd, img_prd1, img_prd2

class Encoder(nn.Module):
    def __init__(self, C):
        super(Encoder, self).__init__()
        ks = 3
        self.conv1 = res_blocks(C, kernel_size=ks, num=3)
        self.conv2 = res_blocks(2*C, kernel_size=ks, num=3)
        self.conv3 = res_blocks(4*C, kernel_size=ks, num=3)
        self.down1 = nn.Sequential(
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 2*C, kernel_size=ks, stride=2),
            res_blocks(2*C, kernel_size=ks, num=3),
        )
        self.down2 = nn.Sequential(
            res_blocks(2*C, kernel_size=ks, num=3),
            conv(2*C, 4*C, kernel_size=ks, stride=2),
            res_blocks(4*C, kernel_size=ks, num=3),
        )
        self.aggE2 = conv(2*C, 2*C, kernel_size=ks, stride=1)
        self.aggE3 = conv(4*C, 4*C, kernel_size=ks, stride=1)

    def forward(self, in1, in2, in3):
        E1 = self.conv1(in1)
        E2 = self.conv2(in2)
        E3 = self.conv3(in3)
        E2 = self.aggE2(self.down1(E1) + E2)
        E3 = self.aggE3(self.down2(E2) + E3)

        return E1+in1, E2+in2, E3+in3

class Decoder(nn.Module):
    def __init__(self, C):
        super(Decoder, self).__init__()
        ks = 3
        self.conv1 = res_blocks(C, kernel_size=ks, num=3)
        self.conv2 = res_blocks(2*C, kernel_size=ks, num=3)
        self.conv3 = res_blocks(4*C, kernel_size=ks, num=3)
        self.up1 = nn.Sequential(
            res_blocks(2*C, kernel_size=ks, num=3),
            upconv(2*C, C),
            res_blocks(C, kernel_size=ks, num=3),
        )
        self.up2 = nn.Sequential(
            res_blocks(4*C, kernel_size=ks, num=3),
            upconv(4*C, 2*C),
            res_blocks(2*C, kernel_size=ks, num=3),
        )
        self.aggD1 = conv(C, C, kernel_size=ks, stride=1)
        self.aggD2 = conv(2*C, 2*C, kernel_size=ks, stride=1)

    def forward(self, in1, in2, in3):
        D1 = self.conv1(in1)
        D2 = self.conv2(in2)
        D3 = self.conv3(in3)
        D2 = self.aggD2(self.up2(D3) + D2)
        D1 = self.aggD1(self.up1(D2) + D1)

        return D1+in1, D2+in2, D3+in3

class EdgeAttention(nn.Module):
    def __init__(self, C):
        super(EdgeAttention, self).__init__()
        self.cat = conv(C*2,C,kernel_size=3,stride=1)
        self.res = res_blocks(C, kernel_size=3, num=2)
        self.res1 = res_blocks(C, kernel_size=3, num=1)
        self.avgpooling1 = nn.AvgPool2d(3, 1, padding=1)
        self.avgpooling2 = nn.AvgPool2d(3, 1, padding=1)
        self.res2 = res_blocks(C, kernel_size=3, num=2)
        self.sig1 = torch.nn.Sigmoid()

    def forward(self, x, edge):
        xx = self.res(x)
        sum = self.res1(edge)
        sum = self.avgpooling1(sum)
        sum = self.cat(torch.cat([sum,x],1))
        sum = self.res2(sum)
        sum = self.avgpooling2(sum)
        map = self.sig1(sum)

        return xx.mul(map)

class model(nn.Module):
    #Best E2D2 with edge
    def __init__(self):
        super(model, self).__init__()
        ks = 3
        C = cfg.NETWORK.CHANNEL
        self.edgenet = EdgeNet()
        self.edgedown1 = nn.Sequential(
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 2 * C, kernel_size=ks, stride=2),
        )
        self.edgedown2 = nn.Sequential(
            res_blocks(2 * C, kernel_size=ks, num=3),
            conv(2 * C, 4 * C, kernel_size=ks, stride=2),
        )
        self.agg_edge1 = conv(2 * C, C, kernel_size=ks, stride=1)
        self.agg_edge2 = conv(4 * C, 2 * C, kernel_size=ks, stride=1)
        self.agg_edge3 = conv(8 * C, 4 * C, kernel_size=ks, stride=1)
        self.agg_skip1 = conv(2 * C, C, kernel_size=ks, stride=1)
        self.agg_skip2 = conv(4 * C, 2 * C, kernel_size=ks, stride=1)
        self.agg_skip3 = conv(8 * C, 4 * C, kernel_size=ks, stride=1)
        self.edgeAttention1 = EdgeAttention(C)
        self.edgeAttention2 = EdgeAttention(2 * C)
        self.edgeAttention3 = EdgeAttention(4 * C)
        # encoder
        self.encoder1 = Encoder(C)
        self.encoder2 = Encoder(C)
        self.decoder1 = Decoder(C)
        self.decoder2 = Decoder(C)
        self.conv1 = conv(3, C, kernel_size=ks, stride=1)
        self.conv2 = conv(3, 2*C, kernel_size=ks, stride=1)
        self.conv3 = conv(3, 4*C, kernel_size=ks, stride=1)



        self.img_prd = nn.Sequential(
            conv(C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 3, kernel_size=ks, stride=1),
        )
        self.img_prd1 = nn.Sequential(
            conv(2*C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 3, kernel_size=ks, stride=1),
        )
        self.img_prd2 = nn.Sequential(
            conv(4*C, C, kernel_size=ks, stride=1),
            res_blocks(C, kernel_size=ks, num=3),
            conv(C, 3, kernel_size=ks, stride=1),
        )


    def forward(self, x):
        downsample1 = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
        downsample2 = F.interpolate(x, scale_factor=0.25, recompute_scale_factor=True)
        Fedge1, edge1, edge2, edge3 = self.edgenet(x)
        Fedge2 = self.edgedown1(Fedge1)
        Fedge3 = self.edgedown2(Fedge2)
        # encoder
        E1_1 = self.conv1(x)
        E2_1 = self.conv2(downsample1)
        E3_1 = self.conv3(downsample2)
        E1_1 = self.agg_edge1(torch.cat([E1_1, Fedge1], 1))
        E2_1 = self.agg_edge2(torch.cat([E2_1, Fedge2], 1))
        E3_1 = self.agg_edge3(torch.cat([E3_1, Fedge3], 1))
        E1_2, E2_2, E3_2 = self.encoder1(E1_1, E2_1, E3_1)
        E1_3, E2_3, E3_3 = self.encoder2(E1_2, E2_2, E3_2)
        # decoder

        D1_1, D2_1, D3_1 = self.decoder1(E1_3, E2_3, E3_3)
        D1_1 = self.edgeAttention1(D1_1, Fedge1)
        D2_1 = self.edgeAttention2(D2_1, Fedge2)
        D3_1 = self.edgeAttention3(D3_1, Fedge3)

        D1_1 = self.agg_skip1(torch.cat([D1_1, E1_2], 1))
        D2_1 = self.agg_skip2(torch.cat([D2_1, E2_2], 1))
        D3_1 = self.agg_skip3(torch.cat([D3_1, E3_2], 1))

        D1_2, D2_2, D3_2 = self.decoder2(D1_1, D2_1, D3_1)

        img_prd = self.img_prd(D1_2)
        img_prd1 = self.img_prd1(D2_2)
        img_prd2 = self.img_prd2(D3_2)
        # para = list(self.conv3_2.parameters())
        # print(para)
        # # len返回列表项个数
        # print(len(para))
        ans = []
        ans.append(img_prd + x)
        ans.append(img_prd1 + downsample1)
        ans.append(img_prd2 + downsample2)
        EDGE = []
        EDGE.append(edge1)
        EDGE.append(edge2)
        EDGE.append(edge3)
        return ans, EDGE

if __name__ == '__main__':
    # cfg.NETWORK.NUM_RESBLOCK = 3
    model = model()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (3, 256, 256))
    #
    # input = torch.randn(1,3,256,256).cuda()
    # flops, params = profile(model, inputs=(input, ))
    # print(flops, params)

    # stat(model, (3, 256, 256))
