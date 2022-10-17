"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import models.networks as networks
from lpips_pytorch import lpips
from util import util
from .EntropyBottleneck import EntropyBottleneck, TextureEntropyBottleneck
from .networks import *


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


class CSEANModel(torch.nn.Module):
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

        if self.opt.train_semantic:
            self.semantic_codec = Codec(self.opt.codec_nc).to(**tensor_kwargs)

        self.texture_entropy = TextureEntropyBottleneck(self.opt.style_nc).to(**tensor_kwargs)
        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            # self.criterionPerc = LPIPS(net_type='alex', version='0.1')
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode='generator', style_code=None):

        input_semantics = data['label'].cuda()
        real_image = data['image'].cuda()
        img_dims = tuple(real_image.size()[1:])
        # print(data['path'], img_dims)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding

        if mode == 'test':
            n_encoder_downsamples = self.netE.n_downsampling_layers
            factor = 2 ** n_encoder_downsamples
            real_image = util.pad_factor(real_image, real_image.size()[2:], factor)
            input_semantics = util.pad_factor(input_semantics, input_semantics.size()[2:], factor)
            input_semantics_onehot_pad = self.to_onehot(input_semantics)  # after padding

        if mode == 'transfer':
            # style_matrix = style_matrix / 1e-2  # 避免量化之后都为0的情况
            # downsampling = 0 in texture entropy, only project to low dimension in channel, no need for padding
            style_code_hat, style_z_hat, texture_length, _ = self.texture_entropy(style_code / 1e-2)
            transfer_img = self.netG(input_semantics_onehot, style_code_hat * 1e-2)
            return transfer_img
        # texture encoder
        style_matrix = self.netE(real_image, input_semantics).cuda()  # (bs,512,19,1)
        # make sure channel is odd

        if mode == 'encode':
            return style_matrix

        # entropy for two layers
        style_matrix = style_matrix / 1e-2  # 避免量化之后都为0的情况
        # downsampling = 0 in texture entropy, only project to low dimension in channel, no need for padding
        style_matrix_hat, style_z_hat, texture_length, latents_decor = self.texture_entropy(style_matrix)
        style_matrix_hat = style_matrix_hat * 1e-2
        style_matrix = style_matrix * 1e-2

        if mode == 'entropy':
            return style_matrix, latents_decor

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W

        if self.opt.train_semantic:
            # enlarge the dynamic range of value
            input_semantics = input_semantics * 10.0
            semantic_code, semantic_hat, semantic_length = self.semantic_codec(input_semantics)
            semantic_hat = semantic_hat / 10.0
            semantic_hat = (torch.round(
                semantic_hat) - semantic_hat).detach() + semantic_hat  # label value should be integer
            semantic_hat = semantic_hat.clamp(0, torch.max(
                input_semantics / 10))  # label value should be within class num range
            semantic_onehot_hat = self.to_onehot(semantic_hat)
            # train nets with decoded semantic map
            decode_image = self.netG(semantic_onehot_hat, style_matrix_hat)
            # compute sem loss first
            sem_mse = torch.mean((semantic_hat - input_semantics / 10) ** 2)
            sem_bpp = semantic_length / (input_semantics.numel() / input_semantics.size(1))
        else:
            # train nets with accurate semantic map
            if mode == 'test':
                decode_image = self.netG(input_semantics_onehot, style_matrix_hat, img_dims)
            else:
                decode_image = self.netG(input_semantics_onehot, style_matrix_hat)

        # loss_sem = self.opt.k_sem * sem_mse + self.opt.lmbda * sem_bpp

        # compute loss
        if mode == 'generator':
            G_losses = self.compute_generator_loss(input_semantics, input_semantics_onehot, real_image, decode_image)

            # if torch.isnan(tex_bpp).any():
            #     raise AssertionError("nan in tex_bpp")

            # bpp = (semantic_length + texture_length) / (real_image.numel()) / real_image.size(1)
            # test only mse
            # loss = loss_mse + self.opt.lmbda * bpp
            if self.opt.train_semantic:
                loss = self.opt.k_lpips * G_losses['lpips'] + self.opt.k_mse * G_losses[
                    'mse'] + self.opt.k_sem * sem_mse + self.opt.k_feat * G_losses['GAN_Feat'] + self.opt.lmbda * (
                               tex_bpp + sem_bpp)
                return loss, sem_mse, tex_bpp, sem_bpp, decode_image, semantic_hat, [G_losses['lpips'], G_losses['mse']]
            else:
                semantic_hat = input_semantics
                if not self.opt.no_vgg_loss:
                    # loss = self.opt.k_lpips * G_losses['lpips'] + self.opt.k_mse * G_losses['mse'] + self.opt.k_latent * \
                    #        G_losses['latent'] + G_losses['GAN_Feat'] + G_losses['VGG'] + self.opt.k_gan * G_losses[
                    #         'GAN'] + self.opt.lmbda * tex_bpp
                    loss = self.opt.k_latent * G_losses['latent'] + G_losses['GAN_Feat'] + G_losses[
                        'VGG'] + self.opt.k_gan * G_losses[
                               'GAN'] + self.opt.lmbda * tex_bpp
                    return loss, tex_bpp, decode_image, semantic_hat, [G_losses['lpips'], G_losses['mse'],
                                                                       G_losses['GAN_Feat'],
                                                                       G_losses['GAN'], G_losses['latent'],
                                                                       G_losses['VGG']]
                else:
                    loss = self.opt.k_lpips * G_losses['lpips'] + self.opt.k_mse * G_losses['mse'] + self.opt.k_latent * \
                           G_losses['latent'] + G_losses['GAN_Feat'] + self.opt.k_gan * G_losses[
                               'GAN'] + self.opt.lmbda * tex_bpp

                    return loss, tex_bpp, decode_image, semantic_hat, [G_losses['lpips'], G_losses['mse'],
                                                                       G_losses['GAN_Feat'],
                                                                       G_losses['GAN'], G_losses['latent']]
        elif mode == 'discriminator':
            D_losses = self.compute_discriminator_loss(input_semantics_onehot, real_image, decode_image)
            loss = sum(D_losses.values()).mean()
            return loss
        elif mode == 'test':
            return decode_image, tex_bpp, style_matrix_hat
        else:
            return decode_image, tex_bpp, style_matrix_hat

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        G_params += list(self.netE.parameters())
        G_params += list(self.texture_entropy.parameters())
        # semantic_codec independent loss
        if self.opt.train_semantic:
            G_params += list(self.semantic_codec.parameters())
        # if opt.use_vae:
        #     G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2  # 默认使用TTUR，G 0.0001，D 0.0004

        optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

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
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt)
        print(netG, netE)
        # netE = networks.define_E(opt) if opt.use_vae else None
        if opt.load_codec:
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            save_path = os.path.join(save_dir, "checkpoint.pth")
            checkpoint = torch.load(save_path)
            self.semantic_codec.load_state_dict(checkpoint["network"])
            # self.texture_entropy.load_state_dict(checkpoint["network"], strict=False)
            print("load semantic codec success")
        # print(self.texture_entropy)
        # print(self.semantic_codec)
        # print(netG)
        # print(self.opt.label_nc,self.opt.semantic_nc)
        if not opt.isTrain or opt.continue_train:
            if opt.train_semantic and not opt.load_codec:
                self.semantic_codec = util.load_network(self.semantic_codec, 'semantic_entropy', opt.which_epoch, opt)
                print("load semantic codec success")
            print(self.texture_entropy)
            self.texture_entropy = util.load_network(self.texture_entropy, 'texture_entropy', opt.which_epoch, opt)
            print("load texture entropy success")
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            print("load netG success")
            # print(netE.named_parameters)
            netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            print("load netE success")
            if opt.GANmode:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                print("load netD success")

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def to_onehot(self, label):
        label_map = label.long().cuda()
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        return input_semantics_onehot

    def preprocess_input(self, data):
        # move to GPU and change data types
        # data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            # data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        if self.opt.label_type == 'edge':
            input_semantics_onehot = data['label']
        else:
            label_map = data['label'].long().cuda()
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics_onehot = torch.cat((input_semantics_onehot, instance_edge_map), dim=1)

        return data['label'], input_semantics_onehot, data['image']

    def compute_generator_loss(self, input_semantics, input_semantics_onehot, real_image, decode_image):
        G_losses = {}

        # if self.opt.use_vae:
        #     G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics_onehot, decode_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.k_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        G_losses['lpips'] = lpips(self.opt, real_image, decode_image)
        if torch.isnan(G_losses['lpips']).any():
            raise AssertionError("nan in loss_lpips")

        G_losses['mse'] = torch.mean((real_image - decode_image) ** 2) * 255 ** 2
        if torch.isnan(G_losses['mse']).any():
            raise AssertionError("nan in loss_mse")

        # L1 latent code loss
        texture_code_real = self.netE(real_image, input_semantics)
        texture_code_fake = self.netE(decode_image, input_semantics)

        G_losses['latent'] = self.criterionFeat(texture_code_fake, texture_code_real).cuda()
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(decode_image, real_image) \
                              * self.opt.lambda_vgg

        return G_losses

    def compute_discriminator_loss(self, input_semantics_onehot, real_image, decode_image):
        D_losses = {}
        # with torch.no_grad():
        # fake_image = self.generate_fake(input_semantic
        # s, input_semantics_onehot, real_image)
        decode_image = decode_image.detach()
        decode_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics_onehot, decode_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

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
            style_matrix = self.netE(real_image).cuda()
        else:
            style_matrix = self.netE(real_image, input_semantics).cuda()
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
