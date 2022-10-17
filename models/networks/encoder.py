"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


def style_extract(feature_map, mask, label_nc):
    x = feature_map
    mask = F.interpolate(mask, size=(x.size()[2], x.size()[3]), mode='nearest')  # (batch_size, 3,128,128)
    mask = mask.round()

    style_matrix = torch.zeros((x.size()[0], x.size()[1], label_nc + 1, 1))  # (batch_size,512,151,1)

    region_list = np.unique(mask.cpu().numpy().astype(int))

    for i in region_list:
        indices = (mask == i)
        sum_indices = torch.sum(indices, dim=(2, 3), keepdim=True)
        ones = torch.ones_like(sum_indices)
        sum_indices = torch.where(sum_indices > 0, sum_indices, ones)
        sum_features = torch.sum(indices * x, dim=(2, 3), keepdim=True)
        style = sum_features / sum_indices
        style = style.view(style.size(0), style.size(1), 1, 1)

        style_matrix[:, :, i:i + 1, :] = style

    # for i in region_list:
    #     # import pdb
    #     # pdb.set_trace()
    #     for j in range(x.size()[0]):
    #         indices = (mask[j:j + 1, :, :, :] == i)
    #         style = torch.sum(indices * x[j:j + 1, :, :, :], dim=(2, 3)) / torch.sum(indices)
    #         style = style.view(1, style.size(1), 1, 1)
    #         style_matrix[j:j+1, :, i:i + 1, :] = style
    if torch.isnan(style_matrix).any():
        raise AssertionError("nan in st_0")

    #     if (style_matrix != style_matrix).any():
    #         assert "Nan error"
    return style_matrix


class BasicBlockDown(nn.Module):
    def __init__(self, inplanes, outplanes, opt, nl_layer=None):
        super().__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.acvtv = nn.LeakyReLU(0.2, inplace=True)
        self.layer1 = norm_layer(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.layer2 = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shortcut_conv = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_0 = self.layer1(self.acvtv(x))
        x_0 = self.avg(self.layer2(self.acvtv(x_0)))
        x_s = self.shortcut_conv(self.avg(x))
        out = x_0 + x_s
        return out


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, opt, nl_layer=None):
        super().__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.acvtv = nn.LeakyReLU(0.2, inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer1 = norm_layer(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.layer2 = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.shortcut_conv = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.up(x)
        x_1 = self.layer1(self.acvtv(x))
        x_1 = self.layer2(self.acvtv(x_1))
        x_s = self.shortcut_conv(x)
        out = x_1 + x_s
        return out


class ResnetEncoder(BaseNetwork):
    """encoder built with resnet block style: 128 channel"""

    def __init__(self, opt):
        super().__init__()
        self.label_nc = opt.label_nc
        ndf = 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.conv_0 = norm_layer(nn.Conv2d(3, ndf, kernel_size=3, stride=1, padding=1, bias=True))  # 256*256
        self.block_down_0 = BasicBlockDown(ndf, ndf * 2, opt)  # 128*128*128
        self.block_down_1 = BasicBlockDown(ndf * 2, ndf * 4, opt)  # 256*64*64
        self.block_up_0 = BasicBlockUp(ndf * 4, ndf * 2, opt)  # 128*128*128
        # self.block_up_1 = BasicBlockUp(ndf * 4, ndf * 8, opt)
        self.actv = nn.Tanh()
        self.avg = nn.AvgPool2d(128)

    def forward(self, x, mask=None):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.conv_0(x)
        x = self.block_down_0(x)
        x = self.block_down_1(x)
        x = self.block_up_0(x)
        x = self.actv(x)

        if mask is None:
            out = self.avg(x)
        else:
            out = style_extract(x, mask, self.label_nc)
        return out


class StyleEncoder(BaseNetwork):
    """
    encoder structure of SEAN
    """

    def __init__(self, opt):
        super().__init__()
        self.label_nc = opt.label_nc
        self.opt = opt
        final_nc = opt.style_nc
        ndf = 32
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.n_downsampling_layers = 2
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.Tconvlayer = norm_layer(nn.ConvTranspose2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, output_padding=1))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 8, final_nc, kw, stride=1, padding=pw))
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        if opt.non_local:
            self.conv_match1 = nn.Conv2d(final_nc, final_nc // 2, kernel_size=1, stride=1, padding=0)
            self.acts1 = nn.PReLU()
            self.conv_match2 = nn.Conv2d(final_nc, final_nc // 2, kernel_size=1, stride=1, padding=0)
            self.g = nn.Conv2d(final_nc, final_nc // 2, kernel_size=1, stride=1, padding=0)
            self.w = norm_layer(nn.Conv2d(final_nc // 2, final_nc, kernel_size=1, stride=1, padding=0))
        # self.actvn1 = nn.Tanh()

    def forward(self, x, mask):
        if not torch.is_tensor(x):
            print("input of Encoder not tensor")
        # if x.size(2) != 256 or x.size(3) != 256:
        #     x = F.interpolate(x, size=(256, 256), mode='bilinear').float()
        if not torch.is_tensor(x):
            print("input of Encoder not tensor")
        try:
            x = self.layer1(x)
        except RuntimeError:
            print("runtime error, print tensor input is ", x)
        else:
            img_dim = tuple(x.size()[1:])
            x = self.layer2(self.actvn(x))
            x = self.layer3(self.actvn(x))
            x = self.Tconvlayer(self.actvn(x))
            x = self.layer4(self.actvn(x))
            # x = self.actvn1(x)  # (batch_size, 512, 128, 128)
        
        if self.opt.non_local:
            x_embed_1 = self.conv_match1(x)  # N*C/2*H*W
            x_embed_2 = self.conv_match2(x)  # N*C/2*H*W
            g_x = self.g(x)  # N*C/2*H*W
            N, C, H, W = x_embed_1.shape
            x_embed_1 = x_embed_1.view(N, C, H * W)
            x_embed_1 = x_embed_1.permute(0, 2, 1)
            x_embed_2 = x_embed_2.view(N, C, H * W)
            g_x = g_x.view(N, C, H * W)
            g_x = g_x.permute(0, 2, 1)
            f = torch.matmul(x_embed_1, x_embed_2)
            f_div = F.softmax(f, dim=-1)

            y = torch.matmul(f_div, g_x)  # N*HW*C
            y = y.view(N, H, W, C).permute(0, 3, 1, 2) # 1,32,128,128
            w_y = self.w(y)
            x = w_y + x

        style_matrix = style_extract(x, mask, self.label_nc)

        return style_matrix  # (bs,512,151,1)


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, True)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
