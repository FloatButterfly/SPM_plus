"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import STYLEResnetBlock as STYLEResnetBlock
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, STYLE
from util import util


# spectralstylesyncbatch3x3

class UnetGenerator(BaseNetwork):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralstyleinstance3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_channel = opt.label_nc + 1
        ndf = 64
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        self.down_0 = spectral_norm(
            nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1))  # 3*256*256-> 64*128*128
        self.norm_0 = STYLE(spade_config_str, ndf, opt.semantic_nc, opt.style_nc)

        self.down_1 = spectral_norm(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))  # 64*128*128 -> 128*64*64
        self.norm_1 = STYLE(spade_config_str, ndf * 2, opt.semantic_nc, opt.style_nc)

        self.down_2 = spectral_norm(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1))  # 128*64*64 -> 256*32*32
        self.norm_2 = STYLE(spade_config_str, ndf * 4, opt.semantic_nc, opt.style_nc)

        self.down_3 = spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))  # 256*32*32-> 512*16*16
        self.norm_3 = STYLE(spade_config_str, ndf * 8, opt.semantic_nc, opt.style_nc)

        self.up_0 = spectral_norm(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1))  # 512*16*16->256*32*32
        self.norm_4 = STYLE(spade_config_str, ndf * 4, opt.semantic_nc, opt.style_nc)

        self.up_1 = spectral_norm(
            nn.ConvTranspose2d(ndf * 8, ndf * 2, kernel_size=4, stride=2, padding=1))  # 256+256 *32*32-> 128*64*64
        self.norm_5 = STYLE(spade_config_str, ndf * 2, opt.semantic_nc, opt.style_nc)

        self.up_2 = spectral_norm(
            nn.ConvTranspose2d(ndf * 4, ndf, kernel_size=4, stride=2, padding=1))  # 128+128 64*64 -> 64*128*128
        self.norm_6 = STYLE(spade_config_str, ndf, opt.semantic_nc, opt.style_nc)

        self.up_3 = spectral_norm(
            nn.ConvTranspose2d(ndf, 3, kernel_size=4, stride=2, padding=1))  # 64*128*128->3*256*256
        # self.norm_7 = STYLE(spade_config_str, 3, opt.semantic_nc, opt.style_nc)

        self.actv_1 = nn.Tanh()

    def forward(self, input, st):
        seg = input
        x = F.interpolate(seg, size=(256, 256), mode='nearest')

        x_0 = self.down_0(self.actv(x))
        x_0 = self.norm_0(x_0, seg, st)

        x_1 = self.down_1(self.actv(x_0))
        x_1 = self.norm_1(x_1, seg, st)

        x_2 = self.down_2(self.actv(x_1))
        x_2 = self.norm_2(x_2, seg, st)

        x_3 = self.down_3(self.actv(x_2))
        x_3 = self.norm_3(x_3, seg, st)

        x_4 = self.up_0(self.actv(x_3))
        x_4 = self.norm_4(x_4, seg, st)

        x_5 = self.up_1(self.actv(torch.cat([x_4, x_2], 1)))
        x_5 = self.norm_5(x_5, seg, st)

        x_6 = self.up_2(self.actv(torch.cat([x_5, x_1], 1)))
        x_6 = self.norm_6(x_6, seg, st)

        out = self.up_3(self.actv(x_6))
        out = self.actv_1(out)

        return out


class STYLEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralstyleinstance3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('less', 'normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf  # 64

        self.sw, self.sh, self.num_upsampling_layers = self.compute_latent_vector_size(opt)

        # if opt.use_vae:
        #     # In case of VAE, we will sample from random z vector
        #     self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        # else:
        #     # Otherwise, we make the network deterministic by starting with
        #     # downsampled segmentation map instead of random z
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = STYLEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = STYLEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = STYLEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = STYLEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = STYLEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = STYLEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = STYLEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = STYLEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2,mode='bicubic')

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        elif opt.num_upsampling_layers == 'less':
            num_up_layers = 3
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)
        # if opt.crop_size == 512 and opt.num_upsampling_layers == '':
        #     sw = opt.crop_size // (2 ** (num_up_layers+1))
        # else:
        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh, num_up_layers

    def forward(self, input, st, img_dims=None):
        seg = input

        if img_dims is None:
            sh, sw = self.sh, self.sw
        else:
            factor = 2 ** self.num_upsampling_layers
            seg = util.pad_factor(seg, seg.size()[2:], factor)
            sh, sw = seg.size()[2] // factor, seg.size()[3] // factor
        # if self.opt.use_vae:
        #     # we sample z from unit normal and reshape the tensor
        #     if z is None:
        #         z = torch.randn(input.size(0), self.opt.z_dim,
        #                         dtype=torch.float32, device=input.get_device())
        #     x = self.fc(z)
        #     x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        # else:
        #     # we downsample segmap and run convolution
        if self.opt.label_type != 'edge':
            x = F.interpolate(seg, size=(sh, sw), mode='nearest')
        else:
            x = F.interpolate(seg, size=(sh, sw), mode='bilinear')

        x = self.fc(x)

        x = self.head_0(x, seg, st)  # 32*32*1024
        x = self.up(x)  # 64*64*1024

        # with open(os.path.join(logdir), 'a') as f:
        #     f.write("after upsample : \n" + x + '\n')

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_0(x, seg, st)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_1(x, seg, st)
            x = self.up(x)

        x = self.up_0(x, seg, st)  # 64*64*512
        x = self.up(x)

        x = self.up_1(x, seg, st)  # 128*128*256
        x = self.up(x)

        x = self.up_2(x, seg, st)  # 256*256*128

        if self.opt.num_upsampling_layers != 'less':
            x = self.up(x)

        x = self.up_3(x, seg, st)  # 256*256*64

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, st)

        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))

        x = F.tanh(x)

        if img_dims is not None:
            x = x[:, :, :img_dims[1], :img_dims[2]]

        return x


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadeinstance3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most', 'less'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', "
                                 "also add one more upsampling + resnet layer at the end of the generator.If 'less', "
                                 "cut one middle resnet block.")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)  # 32*32*1024

        x = self.up(x)  # 64*64*1024

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        # if self.opt.up_middle:
        # x = self.up(x)

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_1(x, seg)
            x = self.up(x)

        x = self.up_0(x, seg)  # 64*64*512
        x = self.up(x)
        x = self.up_1(x, seg)  # 128*128*256
        x = self.up(x)
        x = self.up_2(x, seg)  # 256*256*128

        if self.opt.num_upsampling_layers != 'less':
            x = self.up(x)

        x = self.up_3(x, seg)  # 256*256*64

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(inplace=True)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
