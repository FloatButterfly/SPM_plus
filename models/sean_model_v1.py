# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import models.networks as networks
from lpips_pytorch import lpips
from util import util
from .EntropyBottleneck import EntropyBottleneck, TextureEntropyBottleneck_v3, TextureEntropyBottleneck, TextureEntropyBottleneck_v2, TextureEntropyBottleneck_v4, TextureEntropyBottleneck_v1,TextureEntropyBottleneck_v0, TextureEntropyBottleneck_GSM,TextureEntropyBottleneck_no_hyperprior 
from .networks import *
# from apex import amp


class Codec(nn.Module):
    def __init__(self, width):
        super(Codec, self).__init__()
        self._width = int(width)
        self.encoder = Encoder_GDN(1, self._width, self._width)
        self.decoder = Decoder_GDN(self._width, self._width, 1)
        self.entropy = EntropyBottleneck(self._width)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, z_hat, length = self.entropy(y)
        x_hat = self.decoder(y_hat)
        code = (y_hat, z_hat)
        return code, x_hat, length

    def offset(self):
        return self.entropy.offset


tensor_kwargs = {"dtype": torch.float32, "device": torch.device("cuda:0")}


# +
class CSEANModel_1(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.flag = True if self.opt.semantic_nc % 2 else False
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        if self.opt.semantic_prior:
            self.texture_entropy = TextureEntropyBottleneck_v1(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.semantic_prior_v2:
            self.texture_entropy = TextureEntropyBottleneck_v2(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.semantic_prior_v3:
            self.texture_entropy = TextureEntropyBottleneck_v3(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.semantic_prior_v4:
            self.texture_entropy = TextureEntropyBottleneck_v4(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.adap_qp:
            self.texture_entropy = TextureEntropyBottleneck_v0(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.GSM:
            self.texture_entropy = TextureEntropyBottleneck_GSM(self.opt.style_nc, opt).to(**tensor_kwargs)
        elif self.opt.no_hyper:
            self.texture_entropy = TextureEntropyBottleneck_no_hyperprior(self.opt.style_nc).to(**tensor_kwargs)
        else:
            self.texture_entropy = TextureEntropyBottleneck(self.opt.style_nc, opt).to(**tensor_kwargs)

        self.texture_entropy.init_weights(opt.init_type, opt.init_variance)
        if self.opt.k_cd:
            self.netG, self.netD, self.netE, self.netCD = self.initialize_networks(opt)
        else:
            self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if self.opt.fixed_EGD:
            self.set_requires_grad([self.netE, self.netG, self.netD], False)
        if self.opt.fix_E:
            self.set_requires_grad([self.netE], False)
        if self.opt.fix_G:
            self.set_requires_grad([self.netG], False)
        if self.opt.fix_D:
            self.set_requires_grad([self.netD], False)
        if self.opt.fix_P:
            self.set_requires_grad([self.texture_entropy], False)
        if opt.isTrain:
            # self.criterionPerc = LPIPS(net_type='alex', version='0.1')
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionCD = networks.GANLoss('ls', tensor=self.FloatTensor, opt=self.opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            if self.opt.k_vgg:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            init_loss = self.FloatTensor(1).fill_(0).cuda(self.opt.local_rank)
            self.G_losses = {"mse": init_loss, "L1": init_loss, "lpips": init_loss, "GAN_Feat": init_loss,
                             "GAN": init_loss, "g_coocur_loss": init_loss, "latent": init_loss,
                             "latent2": init_loss, "latent_reg": init_loss, "recon_reg": init_loss, "VGG": init_loss,
                             "latent_reg2": init_loss, "latent_ratio": init_loss,
                             "latent_reg2_l2": init_loss, "bpp":init_loss}
            self.D_losses = {"D_fake": init_loss, "D_real": init_loss, "D_occur": init_loss, "D_occur_fake": init_loss,
                             "D_occur_real": init_loss, "occur_r1": init_loss}

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def val_forward(self, data):
        input_semantic = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantic)
        style_matrix = self.netE(real_image, input_semantic).cuda(self.opt.local_rank)
        print("val forward:*** ", real_image.shape)
        if self.opt.adap_qp:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix)
        elif self.opt.binary_quant:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / (2**self.opt.qp_step))
            q_style_matrix1 = q_style_matrix1 * (2**self.opt.qp_step)
        else:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix1 = q_style_matrix1 * self.opt.qp_step
        
        if self.opt.ds_label:
            H,W = input_semantic.size()[2],input_semantic.size()[3]
            mask = F.interpolate(input_semantic, size=(H//8, W//8), mode='nearest')
            input_semantics_onehot = self.to_onehot(mask)
        elif self.opt.ds_label_2:
            H,W = input_semantic.size()[2],input_semantic.size()[3]
            mask = F.interpolate(input_semantic, size=(H//2, W//2), mode='nearest')
            input_semantics_onehot = self.to_onehot(mask)
        elif self.opt.ds_label_4:
            H,W = input_semantic.size()[2],input_semantic.size()[3]
            mask = F.interpolate(input_semantic, size=(H//4, W//4), mode='nearest')
            input_semantics_onehot = self.to_onehot(mask)
        elif self.opt.ds_label_6:
            H,W = input_semantic.size()[2],input_semantic.size()[3]
            mask = F.interpolate(input_semantic, size=(H//6, W//6), mode='nearest')
            input_semantics_onehot = self.to_onehot(mask)
        
        tex_bpp = texture_length1 / (real_image.numel() / real_image.size(1))
        decode_image = self.netG(input_semantics_onehot, q_style_matrix1)

        return decode_image, tex_bpp, q_style_matrix1

    def transfer_forward(self, data, style_code):
        input_semantic = data['label'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantic)
        if self.opt.adap_qp:
            q_style_code, style_z_hat, texture_length, latents_decor = self.texture_entropy(style_code)
        else:
            q_style_code, style_z_hat, texture_length, _ = self.texture_entropy(style_code / self.opt.qp_step)
        transfer_img = self.netG(input_semantics_onehot, q_style_code * self.opt.qp_step)
        return transfer_img

    def test_forward(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        img_dims = tuple(real_image.size()[1:])

        n_encoder_downsamples = self.netE.n_downsampling_layers
        factor = 2 ** n_encoder_downsamples
        real_image = util.pad_factor(real_image, real_image.size()[2:], factor)
        input_semantics = util.pad_factor(input_semantics, input_semantics.size()[2:], factor)
        input_semantics_onehot_pad = self.to_onehot(input_semantics)  # after padding

        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)

        if self.opt.adap_qp:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(style_matrix)
        elif self.opt.binary_quant:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / (2**self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2**self.opt.qp_step)
        else:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix = q_style_matrix * self.opt.qp_step

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
        decode_image = self.netG(input_semantics_onehot, q_style_matrix, img_dims)

        return real_image, decode_image, tex_bpp, q_style_matrix, latents_decor

    def encode_forward(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        style_matrix = self.netE(real_image, input_semantics)  # (bs,512,19,1)
        # style_matrix,x_layer3 = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)
        return style_matrix

    def swap_prepare(self, data):
        input_semantics1, input_semantics2 = data['label'].chunk(2, dim=0)
        real_image1, real_image2 = data['image'].chunk(2, dim=0)
        input_semantics1, input_semantics2, real_image1, real_image2 = input_semantics1.cuda(
            self.opt.local_rank), input_semantics2.cuda(self.opt.local_rank), real_image1.cuda(
            self.opt.local_rank), real_image2.cuda(self.opt.local_rank)
        input_semantics_onehot1 = self.to_onehot(input_semantics1)  # unpadding

        style_matrix1 = self.netE(real_image1, input_semantics1).cuda(self.opt.local_rank)  # (bs,512,19,1)
        style_matrix2 = self.netE(real_image2, input_semantics2).cuda(self.opt.local_rank)

        q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
            style_matrix1 / self.opt.qp_step)
        q_style_matrix2, style_z_hat2, texture_length2, latents_decor2 = self.texture_entropy(
            style_matrix2 / self.opt.qp_step)
        q_style_matrix1 = q_style_matrix1 * self.opt.qp_step
        q_style_matrix2 = q_style_matrix2 * self.opt.qp_step

        tex_bpp = (texture_length1 + texture_length2) / (
                2 * (real_image1.numel() / real_image1.size(1)))  # nbits / B*H*W

        decode_image1 = self.netG(input_semantics_onehot1, q_style_matrix1)
        decode_image2 = self.netG(input_semantics_onehot1, q_style_matrix2)

        restyle_matrix2 = self.netE(decode_image2, input_semantics1).cuda(self.opt.local_rank)
        q_restyle_matrix2, _, _, _ = self.texture_entropy(restyle_matrix2 / self.opt.qp_step)
        q_restyle_matrix2 = q_restyle_matrix2 * self.opt.qp_step

        restyle_matrix1 = self.netE(decode_image1, input_semantics1).cuda(self.opt.local_rank)
        q_restyle_matrix1, _, _, _ = self.texture_entropy(restyle_matrix1 / self.opt.qp_step)
        q_restyle_matrix1 = q_restyle_matrix1 * self.opt.qp_step

        return tex_bpp, input_semantics_onehot1, real_image1, decode_image1, real_image2, decode_image2, q_style_matrix1, q_style_matrix2, q_restyle_matrix1, q_restyle_matrix2

    def recons_prepare(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)

        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        
        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)
        if torch.isnan(style_matrix).any():
            raise AssertionError("nan in style matrix")
        if self.opt.adap_qp:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(style_matrix)
            if torch.isnan(q_style_matrix).any():
                raise AssertionError("nan in q_style_matrix")
        elif self.opt.binary_quant:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / (2**self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2**self.opt.qp_step)
        else:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix = q_style_matrix * self.opt.qp_step
        
        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
#         print(texture_length.require_grad)
#         print(self.opt.semantic_nc)
#         print(input_semantics_onehot.shape, q_style_matrix.shape)
        if self.opt.ds_label:
            H,W = input_semantics.size()[2],input_semantics.size()[3]
            mask = F.interpolate(input_semantics, size=(H//8, W//8), mode='nearest')
            input_semantics_onehot_decode = self.to_onehot(mask)
        elif self.opt.ds_label_2:
            H,W = input_semantics.size()[2],input_semantics.size()[3]
            mask = F.interpolate(input_semantics, size=(H//2, W//2), mode='nearest')
            input_semantics_onehot_decode = self.to_onehot(mask)
        elif self.opt.ds_label_4:
            H,W = input_semantics.size()[2],input_semantics.size()[3]
            mask = F.interpolate(input_semantics, size=(H//4, W//4), mode='nearest')
            input_semantics_onehot_decode = self.to_onehot(mask)
        elif self.opt.ds_label_6:
            H,W = input_semantics.size()[2],input_semantics.size()[3]
            mask = F.interpolate(input_semantics, size=(H//6, W//6), mode='nearest')
            input_semantics_onehot_decode = self.to_onehot(mask)
        else:
            input_semantics_onehot_decode = input_semantics_onehot
            
        decode_image = self.netG(input_semantics_onehot_decode, q_style_matrix)
#         print("decode shaope:*** ", decode_image.shape)

        restyle_matrix = self.netE(decode_image, input_semantics).cuda(self.opt.local_rank)
        if self.opt.binary_quant:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / (2**self.opt.qp_step))
            q_restyle_matrix = q_restyle_matrix * (2**self.opt.qp_step)
        else:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / self.opt.qp_step)
            q_restyle_matrix = q_restyle_matrix * self.opt.qp_step

        return tex_bpp, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix

    def forward(self, data, mode='generator', regularize=False, style_code=None, lmbda=-1):
        if mode == "val":
            decode_image, tex_bpp, q_style_matrix = self.val_forward(data)
            return decode_image, tex_bpp, q_style_matrix
        elif mode == "test":
            real_image, decode_image, tex_bpp, q_style_matrix, noise_scale = self.test_forward(data)
            return real_image, decode_image, tex_bpp, q_style_matrix, noise_scale
        elif mode == "transfer":
            transfer_img = self.transfer_forward(data, style_code)
            return transfer_img
        elif mode == "encode":
            style_matrix = self.encode_forward(data)
            return style_matrix
        elif self.opt.swap_train:
            tex_bpp, input_semantics_onehot1, real_image1, decode_image1, real_image2, decode_image2, q_style_matrix1, q_style_matrix2, q_restyle_matrix1, q_restyle_matrix2 = self.swap_prepare(
                data)
            if mode == "generator":
                loss, G_losses = self.compute_generator_loss(input_semantics_onehot1, real_image1, decode_image1,
                                                             q_style_matrix1, q_restyle_matrix1, real_image2,
                                                             decode_image2,
                                                             q_style_matrix2, q_restyle_matrix2)
                loss = loss + self.opt.lmbda * tex_bpp
                return loss, tex_bpp, decode_image1, decode_image2, G_losses
            elif mode == 'discriminator':
                loss, D_losses = self.compute_discriminator_loss(input_semantics_onehot1, real_image1, decode_image1,
                                                                 real_image2, decode_image2, regularize=regularize)
                return loss, D_losses
        else:
            tex_bpp, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix = self.recons_prepare(
                data)
            if mode == "generator":
                loss, G_losses = self.compute_generator_loss(input_semantics_onehot, real_image, decode_image,
                                                             q_style_matrix, q_restyle_matrix)
                if self.opt.train_continue_rate and lmbda != -1:
                    loss = loss + lmbda * tex_bpp
                else:
                    loss = loss + self.opt.lmbda * tex_bpp
                    self.G_losses['bpp']=self.opt.lmbda * tex_bpp
                return loss, tex_bpp, decode_image, G_losses
            elif mode == 'discriminator' and not self.opt.fixed_EGD:
                loss, D_losses = self.compute_discriminator_loss(input_semantics_onehot, real_image, decode_image)
                return loss, D_losses

    def latent_regularizer(self, image1, image2, latent_code1, latent_code2):
        pass

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        G_params += list(self.netE.parameters())
        G_params += list(self.texture_entropy.parameters())

        # semantic_codec independent loss
        if self.opt.train_semantic:
            G_params += list(self.semantic_codec.parameters())
        # if opt.use_vae:
        #     G_params += list(self.netE.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2  # 默认使用TTUR，G 0.0001，D 0.0004

        self.optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(beta1, beta2))

#         if opt.isTrain:
        D_params = list(self.netD.parameters())
        self.optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
#             if self.opt.k_cd:
#                 CD_params = list(self.netCD.parameters())
#                 optimizer_CD = torch.optim.AdamW(CD_params, lr=opt.cd_lr, betas=(beta1, beta2))
#                 return optimizer_G, optimizer_D, optimizer_CD
        return self.optimizer_G, self.optimizer_D

    def load_network(net, label, epoch, opt):
        save_filename = '%s_net_%s.pth' % (epoch, label)
        print(save_filename)
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        save_path = os.path.join(save_dir, save_filename)
        weights = torch.load(save_path)
        # print(weights)
        # if label == 'E':
        #     net.module.load_state_dict(weights)
        # else:
        net.load_state_dict(weights)
        return net

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netE, 'E', epoch, self.opt)
        util.save_network(self.texture_entropy, 'texture_entropy', epoch, self.opt)
        optimizer = {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }
        torch.save(optimizer, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pkl'))
        if self.opt.k_cd:
            util.save_network(self.netCD, 'CD', epoch, self.opt)
        if self.opt.train_semantic:
            util.save_network(self.semantic_codec, 'semantic_entropy', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################
    def _param_num(self):
        semantic_codec_sum = sum(p.numel() for p in self.semantic_codec.parameters() if p.requires_grad)
        entroy_sum = sum(p.numel() for p in self.texture_entropy.parameters() if p.requires_grad)
        print("semantic codec trainable parameters: %d\n" % semantic_codec_sum)
        print("texture entropy model trainable parameters: %d\n" % entroy_sum)

    def initialize_networks(self, opt):
        print("********", opt.semantic_nc)
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(opt) if opt.isTrain else None
        self.netE = networks.define_E(opt)
        if self.opt.k_cd:
            netCD = networks.define_CD(opt) if opt.isTrain else None
        # print(netG, netE)
        # netE = networks.define_E(opt) if opt.use_vae else None
        if opt.load_codec:
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            save_path = os.path.join(save_dir, "checkpoint.pth")
            checkpoint = torch.load(save_path)
            self.semantic_codec.load_state_dict(checkpoint["network"])
            # self.texture_entropy.load_state_dict(checkpoint["network"], strict=False)
            print("load semantic codec success")
        if self.opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.create_optimizers(opt)
            # if self.opt.apex:
            #     [self.netG,self.netE,self.netD],[self.optimizer_G, self.optimizer_D]=amp.initialize(models=[self.netG,self.netE,self.netD],optimizers=[self.optimizer_G, self.optimizer_D],opt_level="O1")
        
        if not opt.isTrain or opt.continue_train:
            if opt.train_semantic and opt.load_codec:
                self.semantic_codec = util.load_network(self.semantic_codec, 'semantic_entropy', opt.which_epoch, opt)
                print("load semantic codec success")
            print(self.texture_entropy)
            self.texture_entropy = util.load_network(self.texture_entropy, 'texture_entropy', opt.which_epoch, opt)
            print("load texture entropy success")
            self.netG = util.load_network(self.netG, 'G', opt.which_epoch, opt)
            print("load netG success")
            # print(netE.named_parameters)
            self.netE = util.load_network(self.netE, 'E', opt.which_epoch, opt)
            print("load netE success")
            if opt.GANmode:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
                print("load netD success")
            if opt.resume:
                save_dir = os.path.join(opt.checkpoints_dir, opt.name)
                optim_ckpt = torch.load(os.path.join(save_dir, 'optimizer.pkl'))
                self.optimizer_G.load_state_dict(optim_ckpt['optimizer_G'])
                self.optimizer_D.load_state_dict(optim_ckpt['optimizer_D'])
#             if self.opt.k_cd and opt.loadCD:
#                 netCD = util.load_network(netCD, 'CD', opt.which_epoch, opt)
#                 return netG, netD, netE, netCD
        return self.netG, self.netD, self.netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def to_onehot(self, label):
        label_map = label.long().cuda(self.opt.local_rank)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_().cuda(self.opt.local_rank)
        input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        return input_semantics_onehot

    def preprocess_input(self, data):
        # move to GPU and change data types
        # data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda(self.opt.local_rank)
            # data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda(self.opt.local_rank)

        # create one-hot label map
        if self.opt.label_type == 'edge':
            input_semantics_onehot = data['label']
        else:
            label_map = data['label'].long().cuda(self.opt.local_rank)
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_().cuda(self.opt.local_rank)
            input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics_onehot = torch.cat((input_semantics_onehot, instance_edge_map), dim=1)

        return data['label'], input_semantics_onehot, data['image']

    def compute_generator_loss(self, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix,
                               real_image2=None,
                               decode_image2=None, q_style_matrix2=None, q_restyle_matrix2=None):
        # G_losses = {}
        pred_fake, pred_real = self.discriminate(
                input_semantics_onehot, decode_image, real_image)
        if self.opt.k_gan:
            self.G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                     for_discriminator=False)
        if self.opt.k_latent:
            self.G_losses['latent'] = self.criterionL1(q_style_matrix, q_restyle_matrix)
        if self.opt.k_latent2:
            self.G_losses['latent2'] = self.criterionL1(q_style_matrix2, q_restyle_matrix2)
        if self.opt.k_L1:
            self.G_losses['L1'] = self.criterionL1(real_image, decode_image)
        if self.opt.k_latent_reg:
            if not self.opt.batchSize % 2 and not self.opt.swap_train:
                decode_img1 = decode_image[:self.opt.batchSize // 2]
                decode_img2 = decode_image[self.opt.batchSize // 2:]
                q_sty1 = q_style_matrix[:self.opt.batchSize // 2]
                q_sty2 = q_style_matrix[self.opt.batchSize // 2:]
                self.G_losses['latent_reg'] = -self.criterionL1(decode_img1, decode_img2) / (
                        2 - self.criterionL1(q_sty1, q_sty2))
            else:
                self.G_losses['latent_reg'] = -self.criterionL1(decode_image2, decode_image) / (
                        2 - self.criterionL1(q_style_matrix2, q_style_matrix))
        if self.opt.k_latent_reg2:
            if not self.opt.batchSize % 2:
                real_img1 = real_image[:self.opt.batchSize // 2]
                real_img2 = real_image[self.opt.batchSize // 2:]
                q_sty1 = q_style_matrix[:self.opt.batchSize // 2]
                q_sty2 = q_style_matrix[self.opt.batchSize // 2:]
                self.G_losses['latent_reg2'] = -self.criterionL1(real_img1, real_img2) / (
                        2 - self.criterionL1(q_sty1, q_sty2))
        if self.opt.k_latent_reg2_l2:
            if not self.opt.batchSize % 2:
                real_img1 = real_image[:self.opt.batchSize // 2]
                real_img2 = real_image[self.opt.batchSize // 2:]
                q_sty1 = q_style_matrix[:self.opt.batchSize // 2]
                q_sty2 = q_style_matrix[self.opt.batchSize // 2:]
                self.G_losses['latent_reg2_l2'] = -self.criterionL2(real_img1, real_img2) / (
                        2 - self.criterionL1(q_sty1, q_sty2))
        if self.opt.k_latent_ratio:
            if not self.opt.batchSize % 2:
                real_img1 = real_image[:self.opt.batchSize // 2]
                real_img2 = real_image[self.opt.batchSize // 2:]
                q_sty1 = q_style_matrix[:self.opt.batchSize // 2]
                q_sty2 = q_style_matrix[self.opt.batchSize // 2:]
                self.G_losses['latent_ratio'] = ((self.criterionL1(real_img1, real_img2)+ self.opt.cont*self.opt.tar_ratio) / (
                        self.criterionL1(q_sty1, q_sty2)+self.opt.cont)-self.opt.tar_ratio)**2
        if self.opt.k_recon_reg:
            self.G_losses['recon_reg'] = self.criterionL1(real_image, decode_image) / (
                    2 - self.criterionL1(q_style_matrix, q_restyle_matrix))
        if self.opt.k_feat:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0).cuda(self.opt.local_rank)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionL1(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.k_feat / num_D
            self.G_losses['GAN_Feat'] = GAN_Feat_loss
        if self.opt.k_lpips:
            self.G_losses['lpips'] = lpips(self.opt, real_image, decode_image)
            # if torch.isnan(G_losses['lpips']).any():
            #     raise AssertionError("nan in loss_lpips")
        if self.opt.k_mse:
            self.G_losses['mse'] = torch.mean((real_image - decode_image) ** 2) * 255 ** 2
        # if torch.isnan(G_losses['mse']).any():
        #     raise AssertionError("nan in loss_mse")
        if self.opt.k_vgg:
            self.G_losses['VGG'] = self.criterionVGG(decode_image, real_image)

        if self.opt.k_cd:
            fake_patch = util.patchify_image(decode_image2, self.opt.n_crop)
            ref_patch = util.patchify_image(real_image2, self.opt.ref_crop * self.opt.n_crop)
            # print(self.netCD)
            fake_patch_pred, _ = self.netCD(fake_patch, ref_patch, ref_batch=self.opt.ref_crop)
            if self.opt.GAN_CD == 'wgan':
                self.G_losses['g_coocur_loss'] = g_nonsaturating_loss(fake_patch_pred)
            # try other gan: ls hinge
            elif self.opt.GAN_CD == 'lsgan':
                self.G_losses['g_coocur_loss'] = self.criterionCD(fake_patch_pred, True,
                                                                  for_discriminator=False)

        loss = self.opt.k_latent * self.G_losses['latent'] + \
               self.opt.k_feat * self.G_losses['GAN_Feat'] + \
               self.opt.k_vgg * self.G_losses['VGG'] + \
               self.opt.k_gan * self.G_losses['GAN'] + \
               self.opt.k_cd * self.G_losses['g_coocur_loss'] + \
               self.opt.k_latent2 * self.G_losses['latent2'] + \
               self.opt.k_latent_reg * self.G_losses['latent_reg'] + \
               self.opt.k_recon_reg * self.G_losses['recon_reg'] + \
               self.opt.k_L1 * self.G_losses['L1'] + \
               self.opt.k_latent_reg2 * self.G_losses['latent_reg2'] + \
               self.opt.k_latent_reg2_l2 * self.G_losses['latent_reg2_l2']+ \
               self.opt.k_latent_ratio * self.G_losses['latent_ratio']

        return loss, self.G_losses

    def compute_discriminator_loss(self, input_semantics_onehot, real_image, decode_image, real_image2=None,
                                   decode_image2=None, regularize=False):
        # D_losses = {}
        # with torch.no_grad():
        # fake_image = self.generate_fake(input_semantic
        # s, input_semantics_onehot, real_image)
        decode_image = decode_image.detach()
        decode_image.requires_grad_()

        if self.opt.k_gan:
            pred_fake, pred_real = self.discriminate(
                input_semantics_onehot, decode_image, real_image)

            self.D_losses['D_fake'] = self.criterionGAN(pred_fake, False,
                                                        for_discriminator=True)
            self.D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                        for_discriminator=True)
        if self.opt.k_cd:
            real_image2.requires_grad = True
            fake_patch = util.patchify_image(decode_image2, self.opt.n_crop)
            real_patch = util.patchify_image(real_image2, self.opt.n_crop)
            ref_patch = util.patchify_image(real_image2, self.opt.ref_crop * self.opt.n_crop)

            fake_patch_pred, ref_input = self.netCD(fake_patch, ref_patch, ref_batch=self.opt.ref_crop)
            real_patch_pred, _ = self.netCD(real_patch, ref_input=ref_input)

            if self.opt.GAN_CD == 'wgan':
                d_coocur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)
                self.D_losses['D_occur'] = d_coocur_loss

            elif self.opt.GAN_CD == 'lsgan':
                self.D_losses['D_occur_fake'] = self.criterionCD(fake_patch_pred, False,
                                                                 for_discriminator=True)
                self.D_losses['D_occur_real'] = self.criterionCD(real_patch_pred, True,
                                                                 for_discriminator=True)
                # d_coocur_loss = torch.mean(D_losses['D_occur_Fake'] + D_losses['D_occur_real'])
            if regularize:
                cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)
                self.D_losses["occur_r1"] = self.opt.cooccur_r1 / 2 * cooccur_r1_loss * self.opt.d_reg_every
                # r1_loss_sum += 0 * real_patch_pred[0, 0]

        loss = sum(self.D_losses.values()).mean()
        return loss, self.D_losses

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, input_semantics_onehot, real_image):
        # no need
        # z = None
        # KLD_loss = None
        # if self.opt.use_vae:
        #     z, mu, logvar = self.encode_z(real_image)
        #     if compute_kld_loss:
        #         KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        #
        # import pdb
        # pdb.set_trace()
        # with torch.no_grad():
        if self.opt.label_type == 'edge':
            style_matrix = self.netE(real_image).cuda(self.opt.local_rank)
        else:
            style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)
        fake_image = self.netG(input_semantics_onehot, style_matrix)
        # print("G, E finished!")

        # assert (not compute_kld_loss) or self.opt.use_vae, \
        #     "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image

        # Given fake and real image, return the prediction of discriminator

    # for each fake and real image.

    def discriminate(self, input_semantics_onehot, fake_image, real_image):
        fake_concat = torch.cat([input_semantics_onehot, fake_image], dim=1)
        real_concat = torch.cat([input_semantics_onehot, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
