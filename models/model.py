import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

class Generator(nn.Module):
    def __init__(self, ngf=32):
        super(Generator, self).__init__()
        # input is 3 x 32 x 128 x 128  (duplicated by 3 x 1 x 128 x 128)

        self.downConv1 = nn.Conv3d(5, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False)
        self.downConv2 = nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False)
        self.downConv3 = nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False)
        self.downConv4 = nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
        self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

        # get
        self.downBN2 = nn.BatchNorm3d(ngf * 2)
        self.downBN3 = nn.BatchNorm3d(ngf * 4)
        self.downBN4 = nn.BatchNorm3d(ngf * 8)
        self.downBN5 = nn.BatchNorm3d(ngf * 16)
        self.relu = nn.ReLU(inplace = True)

        self.upConv1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
        self.upConv2 = nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
        self.upConv3 = nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.upConv4 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.upConv5 = nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.upConv6 = nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)


        self.upBN1 = nn.BatchNorm3d(ngf * 16)
        self.upBN2 = nn.BatchNorm3d(ngf * 8)
        self.upBN3 = nn.BatchNorm3d(ngf * 4)
        self.upBN4 = nn.BatchNorm3d(ngf * 2)
        self.upBN5 = nn.BatchNorm3d(ngf * 1)

        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        downx1 = self.downConv1(x)
        downx2 = self.downConv2(downx1)
        downx2 = self.downBN2(downx2)
        downx2 = self.lrelu(downx2)
        downx3 = self.downConv3(downx2)
        downx3 = self.downBN3(downx3)
        downx3 = self.lrelu(downx3)
        downx4 = self.downConv4(downx3)
        downx4 = self.downBN4(downx4)
        downx4 = self.lrelu(downx4)
        downx5 = self.downConv5(downx4)
        downx5 = self.downBN5(downx5)
        downx5 = self.lrelu(downx5)
        downx6 = self.downConv6(downx5)

        upx1 = self.upConv1(downx6)
        upx1 = self.upBN1(upx1)
        upx1 = self.relu(upx1)
        upx1 = downx5 + upx1

        upx2 = self.upConv2(upx1)
        upx2 = self.upBN2(upx2)
        upx2 = self.relu(upx2)
        upx2 = downx4 + upx2

        upx3 = self.upConv3(upx2)
        upx3 = self.upBN3(upx3)
        upx3 = self.relu(upx3)
        upx3 = downx3 + upx3

        upx4 = self.upConv4(upx3)
        upx4 = self.upBN4(upx4)
        upx4 = self.relu(upx4)
        upx4 = downx2 + upx4

        upx5 = self.upConv5(upx4)
        upx5 = self.upBN5(upx5)
        upx5 = self.relu(upx5)
        upx5 = downx1 + upx5

        upx6 = self.upConv6(upx5)
        upx6 = self.tanh(upx6)

        return upx6

class Discriminator(nn.Module):
    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 32 x 256 x 256
            nn.Conv3d(5, ndf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 64 x 64
            nn.Conv3d(ndf, ndf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 32 x 32
            nn.Conv3d(ndf *2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 16 x 16
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 8 x 8
            nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 16),
            nn.LeakyReLU(inplace=True),
            # (ndf*16) x 2 x 4 x 4
            nn.Conv3d(ndf * 16, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
