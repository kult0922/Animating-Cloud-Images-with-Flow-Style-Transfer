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
import time
from PIL import ImageFile
from PIL import Image
from torch.utils.data import DataLoader
from models.model import Generator, Discriminator
from utils.utils import cal_optical_flow, weights_init, make_G_input, make_D_input, save_video
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

print(opt)

driving_video_loader = make_test_dataloader(opt)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator(ngf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
print(netG)

# load  model
netG.load_state_dict(torch.load(G_path))
print('load %s !' % G_path)

# source image
source_image = Image.open(opt.sourceImage).convert('RGB')
source_image_transform = transforms.Compose([
                   transforms.Resize( (opt.imageSize, opt.imageSize) ),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
source_image = source_image_transform(source_image).to(device)
source_image = source_image.unsqueeze(1).unsqueeze(0).repeat(1, 1, num_frame, 1, 1)

# driving video
for i, testdata in enumerate(driving_video_loader, 0):
    driving_video = testdata[0].to(device)
    driving_flow = cal_optical_flow(testdata[0]).to(device)
    optical_flow_color = cal_optical_flow(testdata[0], color=True).to(device)
    break

netG.eval()
with torch.no_grad():
    generated_video = netG(torch.cat((source_image, driving_flow), 1))

    result = torch.cat((source_image, driving_video, optical_flow_color, generated_video), 4).permute(2,0,1,3,4)

    source_img_name = os.path.basename(opt.sourceImage)
    driving_video_name = os.path.basename(opt.drivingVideo)
    out_test_path = os.path.join('./checkpoints', opt.checkpoint, 'test', source_img_name + '-' + driving_video_name)
    os.makedirs(out_test_path, exist_ok=True)

    # save output
    for t in range(opt.nframes):
        vutils.save_image(result[t],
        '%s/result_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 1)

