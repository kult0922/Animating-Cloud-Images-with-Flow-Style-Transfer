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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from PIL import ImageFile
from PIL import Image

from torch.utils.data import DataLoader
from models.model import Generator, Discriminator
from utils.utils import cal_optical_flow, weights_init, make_G_input, make_D_input, save_video
from options.train_options import TrainOptions
from data.dataloader import make_dataloader

ImageFile.LOAD_TRUNCATED_IMAGES = True
opt = TrainOptions().parse()
localtime = time.asctime( time.localtime(time.time()) )

start_time = time.time()

train_loader, valid_loader = make_dataloader(opt)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

# Create the generator
netG = Generator().to(device)
if (device.type == 'cuda') and (opt.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(opt.ngpu)))
netG.apply(weights_init)
print(netG)

# Create the Discriminator
netD = Discriminator().to(device)
if (device.type == 'cuda') and (opt.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(opt.ngpu)))
netD.apply(weights_init)
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()
l1_loss = nn.L1Loss(reduction='sum')

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.9))

###################### TRAINING LOOP #########################

iters = 0

# the case of coitinue train
if(opt.startEpoch != 0):
    G_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netG_epoch_' + str(opt.startEpoch) + '.pth')
    D_path = os.path.join('./checkpoints', opt.checkpoint, 'weight', 'netD_epoch_' + str(opt.startEpoch) + '.pth')
    netG.load_state_dict(torch.load(G_path))
    netD.load_state_dict(torch.load(D_path))

    print('load %s !' % G_path)
    print('load %s !' % D_path)

# test data for demo during trainig
for i, testdata in enumerate(valid_loader, 0):
    optical_flow = cal_optical_flow(testdata[0]).to(device)
    G_test_demo_input = make_G_input(testdata[0].to(device), optical_flow, opt.nframes)
    break

print('training start')
print(localtime)
print(opt)

for epoch in range(opt.startEpoch, opt.epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):
        # train data for demo during trainig
        if (iters == 0):
            optical_flow = cal_optical_flow(data[0]).to(device)
            G_train_demo_input = make_G_input(data[0].to(device), optical_flow, opt.nframes)

        # extract optical flow
        optical_flow = cal_optical_flow(data[0]).to(device)

        # Discriminator train { maximize log(D(x)) + log(1 - D(G(z))) }
        netD.zero_grad()
        real_data = data[0].to(device)

        G_input = make_G_input(data[0].to(device), optical_flow, opt.nframes)
        D_input_real = torch.cat((real_data, optical_flow), 1) # true
        D_input_fake = torch.cat((netG(G_input), optical_flow), 1) # false fake を通して勾配がGに伝わらないようにdetach

        output = netD(D_input_real).view(-1)
        label = torch.full((opt.batchSize,), real_label, device=device)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        fake = netG(G_input)
        label.fill_(fake_label)
        output = netD(D_input_fake.detach()).view(-1)
        errD_fake = criterion(output, label)

        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Genarator train { maximize log(D(G(z))) }
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(D_input_fake).view(-1)
        errG_adv = criterion(output, label)
        errL1 = l1_loss(fake, real_data) / ( opt.batchSize * opt.nframes * opt.imageSize * opt.imageSize )
        errG = errG_adv + errL1
        errG.backward()
        optimizerG.step()

        # Output training status
        if i % 100 == 0:
            print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f  L1loss: %.4f' % (epoch + 1, opt.epochs, i, len(train_loader), errD.item(), errG.item(), errL1.item()))

        iters += 1

    # make save dirs
    out_test_demo_path = os.path.join('./checkpoints/', opt.checkpoint, 'video', 'epoch_' + str(epoch + 1), 'test')
    out_train_demo_path = os.path.join('./checkpoints/', opt.checkpoint, 'video', 'epoch_' + str(epoch + 1), 'train')
    os.makedirs(out_train_demo_path, exist_ok=True)
    os.makedirs(out_test_demo_path, exist_ok=True)

    # demo
    with torch.no_grad():
        # train data
        fake_train_demo = netG(G_train_demo_input).permute(2,0,1,3,4)
        save_video(fake_train_demo, opt.nframes, out_train_demo_path, 'generated')
        # test data
        fake_test_demo = netG(G_test_demo_input).permute(2,0,1,3,4)
        save_video(fake_test_demo, opt.nframes, out_test_demo_path, 'generated')

    if (epoch + 1) % 5 == 0:
        # save weight
        model_dir = os.path.join('./checkpoints', opt.checkpoint, 'weight')
        os.makedirs(model_dir, exist_ok=True)
        model_G_name = 'netG_epoch_' + str(epoch + 1) + '.pth'
        model_D_name = 'netD_epoch_' + str(epoch + 1) + '.pth'
        torch.save(netG.state_dict(), os.path.join(model_dir, model_G_name))
        torch.save(netD.state_dict(), os.path.join(model_dir, model_D_name))

    # epoch end
    now_time = time.time()
    elapsed_time = int(now_time - start_time)
    hour_minute_time = divmod(elapsed_time, 3600)
    print('---TIME---')
    print('%d hour %d minute' % (hour_minute_time[0], hour_minute_time[1] / 60))

