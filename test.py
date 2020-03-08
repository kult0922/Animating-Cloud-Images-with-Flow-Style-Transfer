from __future__ import print_function

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
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from PIL import ImageFile
from PIL import Image

#from video_folder import VideoFolder
from torch.utils.data import DataLoader
from models.model import FLOW_GAN_G, FLOW_GAN_D
from utils.utils import cal_optical_flow, cal_lucas_flow
from options.test_options import TestOptions
from data.dataloader import make_dataloader

opt = TestOptions().parse()

localtime = time.asctime( time.localtime(time.time()) )

# parameters
ngpu = 1
ngf = 32
ndf = 32
batch_size = opt.batchSize
num_frame = opt.nframes
frame_size = opt.imageSize
start_epoch = opt.startEpoch


G_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netG_epoch_' + str(opt.startEpoch) + '.pth')
D_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netD_epoch_' + str(opt.startEpoch) + '.pth')

###################### START #########################
print('\n start new program! ')
print(opt)

train_loader, valid_loader = make_dataloader(opt)
print('validloader size: ' + str(len(valid_loader)))

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = FLOW_GAN_G(ngf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
print(netG)

# Create the Discriminator
netD = FLOW_GAN_D(ndf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
print(netD)
###################### TEST #########################

# load  model
netG.load_state_dict(torch.load(G_path))
netD.load_state_dict(torch.load(D_path))

print('load %s !' % G_path)
print('load %s !' % D_path)

# input image
for i, testdata in enumerate(valid_loader, 0):
    G_test = testdata[0].to(device)
    ground_true = G_test.clone()
    G_test = G_test[:,:,0,:,:]
    G_test = G_test.unsqueeze(2).repeat(1, 1, num_frame, 1, 1)
    break

# target flow
for i, testdata in enumerate(valid_loader, 0):
    G_test_flow = testdata[0].to(device)
    optical_flow_test = cal_optical_flow(testdata[0]).to(device)
    G_test = torch.cat((G_test, optical_flow_test), 1)
    break

print('\n start new program! ')

with torch.no_grad():
    # test データから生成
    fake_test = netG(G_test)
    test_fake_s1 = fake_test.permute(2, 0, 1, 3, 4)
    ground_true = ground_true.permute(2, 0, 1, 3, 4)
    G_test_flow = G_test_flow.permute(2, 0, 1, 3, 4)
    out_test_path = os.path.join('./checkpoints', opt.checkpoint, 'test', opt.name)
    os.makedirs(out_test_path, exist_ok=True)

    # save output, ground trueth, flowvideo
    for t in range(num_frame):
        vutils.save_image(test_fake_s1[t],
        '%s/generate_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 8)

    for t in range(num_frame):
        vutils.save_image(ground_true[t],
        '%s/ground_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 8)

    for t in range(num_frame):
        vutils.save_image(G_test_flow[t],
        '%s/flow_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 8)

print('###################### END ##########################')
