"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util.util as util
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from .base_network import ResBlock, ConvLayer, EqualLinear,FusedLeakyReLU


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, inplace=True)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, inplace=True)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class CoocurDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        channel = opt.coor_channel
        size = opt.crop_size  # 256
        kw = 1
        padw = int(np.ceil((kw - 1.0) / 2))
        encoder = [nn.Conv2d(3, channel, kernel_size=kw, stride=1, padding=padw)]
        encoder.append(FusedLeakyReLU(channel))
        ch_multiplier = (2, 4, 8, 12, 12, 24)
        downsample = (True, True, True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(nn.Conv2d(in_ch, channel * 12, k_size))
        encoder.append(FusedLeakyReLU(channel*12))
        self.encoder = nn.Sequential(*encoder)
        self.linear = nn.Sequential(
            EqualLinear(channel * 12 * 2 * feat_size, channel * 64, activation="fused_lrelu"),
            EqualLinear(channel * 64, channel * 64, activation="fused_lrelu"),
            EqualLinear(channel * 64, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, 1),
        )

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        # print(self.encoder)
        # import pdb
        # pdb.set_trace()
        out_input = self.encoder(input)
        if ref_input is None:
            ref_input = self.encoder(reference)
            _, c, h, w = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, c, h, w)
            ref_input = ref_input.mean(1)
        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input
