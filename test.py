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
from options.test_options import TestOptions
from data.dataloader import make_dataloader
from models.model import Generator, Discriminator
from utils.utils import calc_optical_flow, weights_init, make_G_input, make_D_input, save_video

opt = TestOptions().parse()

localtime = time.asctime( time.localtime(time.time()) )

G_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netG_epoch_' + str(opt.startEpoch) + '.pth')

print(opt)

train_loader, valid_loader = make_dataloader(opt)
print('validloader size: ' + str(len(valid_loader)))

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

# Create the generator
netG = Generator().to(device)
if (device.type == 'cuda') and (opt.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(opt.ngpu)))
print(netG)

# load  model
netG.load_state_dict(torch.load(G_path))
print('load %s !' % G_path)

# input data
if opt.mode == 'transfer':
    for i, testdata in enumerate(valid_loader, 0):
        source_image = testdata[0].to(device)
        source_image = source_image[:,:,0,:,:]
        source_image = source_image.unsqueeze(2).repeat(1, 1, opt.nframes, 1, 1)
        break

    for i, testdata in enumerate(valid_loader, 0):
        driving_video = testdata[0].to(device)
        optical_flow = calc_optical_flow(testdata[0]).to(device)
        optical_flow_color = calc_optical_flow(testdata[0], color=True).to(device)
        break

if opt.mode == 'reconstruction':
    for i, testdata in enumerate(valid_loader, 0):
        source_image = testdata[0].to(device)
        driving_video = testdata[0].to(device)
        source_image = source_image[:,:,0,:,:]
        source_image = source_image.unsqueeze(2).repeat(1, 1, opt.nframes, 1, 1)
        optical_flow = calc_optical_flow(testdata[0]).to(device)
        optical_flow_color = calc_optical_flow(testdata[0], color=True).to(device)
        break

netG.eval()
with torch.no_grad():
    generated_video = netG(torch.cat((source_image, optical_flow), 1))
    out_test_path = os.path.join('./checkpoints', opt.checkpoint, 'test', opt.name)
    os.makedirs(out_test_path, exist_ok=True)

    result = torch.cat((source_image, driving_video, optical_flow_color, generated_video), 4).permute(2,0,1,3,4)
    generated_video = generated_video.permute(2,0,1,3,4)

    # save output
    for t in range(opt.nframes):
        vutils.save_image(result[t],
        '%s/result_frame_%03d.png'
        % (out_test_path, t), normalize=True,
        nrow = 1)

