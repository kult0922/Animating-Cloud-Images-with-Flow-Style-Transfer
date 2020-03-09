import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

def calc_optical_flow(data, color=False):
    data = data.numpy().transpose(0, 2, 3, 4, 1)
    if (color):
        flow_data = np.zeros_like(data)
    else:
        flow_data = np.zeros_like(data)
        flow_data = np.delete(flow_data, 0, 4)

    for i, video in enumerate(data, 0):

        for j in range(video.shape[0] - 1): # oprtical flow frame number = original_frame number - 1

            # to gray scale
            prev_frame = cv2.cvtColor(video[j], cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(video[j + 1], cv2.COLOR_BGR2GRAY)
            # 255
            prev_frame = (prev_frame + 1) * 255 / 2
            next_frame = (next_frame + 1) * 255 / 2
            # hsv setting
            hsv = np.zeros_like(video[j])
            hsv[..., 1] = 255

            # optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float64)
            if (color):
                # visualize process
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag.astype(np.float32)
                ang.astype(np.float32)
                hsv[...,0] = ang * 360 / np.pi / 2
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                flow_data[i, j] = rgb
            else:
                flow_data[i, j] = flow

    # tensor format
    flow_data = flow_data.transpose(0, 4, 1, 2, 3)
    flow_data = flow_data.astype(np.float32)

    if color:
        ## reglalization [-1, 1]
        mini = flow_data.min()
        maxi = flow_data.max()
        flow_data = flow_data - mini
        flow_data = flow_data / (maxi - mini) * 2 - 1

    else:
        ## reglalization [-1, 1]
        flow_data = flow_data / flow_data.shape[-1]

    return torch.from_numpy(flow_data)

def make_G_input(video, flow, num_frames):
    video = video[:,:,0,:,:]
    video = video.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
    return torch.cat((video, flow), 1)

def make_D_input(video, flow):
    return

def numpy2image(img):
    img = (img + 1) * 255
    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_video(video, num_frames, save_dir, name):
    for t in range(num_frames):
        frame_name = name + '_frame_%03d.png' % (t)
        #import pdb;pdb.set_trace()
        vutils.save_image(video[t], os.path.join(save_dir, frame_name), normalize=True, nrow = 8)

    '''
def save_video(video, dir_path, file_name):
    for t in range(video.size(0)):
        vutils.save_image(video[t],
        '%s/' + file_name + '_frame_%03d.png'
        % (dir_path, t), normalize=True,
        nrow = 8)
        fn = '{:0=3}'.format(t)
        vutils.save_image(video[t],
        os.path.join(dir_path, file_name + '_frame_' + fn +'.png'),
        normalize=True,
        nrow = 8)
'''
