import sys
import argparse
import os
import torch
import models
import pickle
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        #parser = argparse.ArgumentParser()
        parser.add_argument('--dataroot', default = '../dataset/sky_cloud', help='path to dataset')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        parser.add_argument('--batchSize', type=int, default=32)
        parser.add_argument('--imageSize', type=int, default=128, help='the height and width of the input image to network')
        parser.add_argument('--nframes', type=int, default=32, help='number of frames in each video clip')
        parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
        parser.add_argument('--netG', default = './model/netG_epoch15.pth', help='path to netG')
        parser.add_argument('--netD', default = './model/netD_epoch15.pth', help='path to netD')
        parser.add_argument('--startEpoch', type=int, default=0, help='number of starting epoch')
        parser.add_argument('--checkpoint', default='tmp', help='project name')
        parser.add_argument('--name', default = 'tmp', help='test project name')
        parser.add_argument('--noShuffle', action='store_true', help='do nott shuffle dataset')
        opt = parser.parse_args()

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join('./checkpoints/', opt.checkpoint)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        self.opt = opt
        return self.opt
