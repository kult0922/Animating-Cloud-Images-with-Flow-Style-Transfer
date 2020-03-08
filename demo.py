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
from data.dataloader import make_test_dataloader

opt = TestOptions().parse()

localtime = time.asctime( time.localtime(time.time()) )

# parameters
ngpu = 1
ngf = 32
ndf = 32
batch_size = 1
num_frame = opt.nframes
frame_size = opt.imageSize
start_epoch = opt.startEpoch


G_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netG_epoch_' + str(opt.startEpoch) + '.pth')
D_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netD_epoch_' + str(opt.startEpoch) + '.pth')

###################### START #########################
print('\n start new program! ')
print(opt)

driving_video_loader = make_test_dataloader(opt)
print('validloader size: ' + str(len(driving_video_loader)))

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

print('load %s !' % G_path)

# source image
source_img = Image.open(opt.sourceImage).convert('RGB')
source_img_transform = transforms.Compose([
                   transforms.Resize( (opt.imageSize, opt.imageSize) ),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
source_img = source_img_transform(source_img).to(device)
source_img = source_img.unsqueeze(1).unsqueeze(0).repeat(1, 1, num_frame, 1, 1)

# driving video
for i, testdata in enumerate(driving_video_loader, 0):
    driving_video = testdata[0].to(device)
    driving_flow = cal_optical_flow(testdata[0]).to(device)
    driving_flow_color = cal_optical_flow(testdata[0], color=True).to(device)
    G_input = torch.cat((source_img, driving_flow), 1)
    break

print('\n start new program! ')

netG.eval()
with torch.no_grad():
    # test データから生成
    fake_test = netG(G_input)
    test_fake_s1 = fake_test.permute(2, 0, 1, 3, 4)
    driving_flow_color = driving_flow_color.permute(2, 0, 1, 3, 4)
    driving_video = driving_video.permute(2, 0, 1, 3, 4)
    source_img_name = os.path.basename(opt.sourceImage)
    driving_video_name = os.path.basename(opt.drivingVideo)
    out_test_path = os.path.join('./checkpoints', opt.checkpoint, 'test', source_img_name + '-' + driving_video_name)
    os.makedirs(out_test_path, exist_ok=True)

    # save output, ground trueth, flowvideo
    vutils.save_image(source_img.permute(2, 0, 1, 3, 4)[0], '%s/source_img.png' % (out_test_path), normalize=True, nrow = 1)

    for t in range(num_frame):
        vutils.save_image(driving_video[t],
        '%s/driving_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 1)

    for t in range(num_frame):
        vutils.save_image(test_fake_s1[t],
        '%s/generate_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 8)

    for t in range(num_frame):
        vutils.save_image(driving_flow_color[t],
        '%s/flow_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 8)

print('###################### END ##########################')
