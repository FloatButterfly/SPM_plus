# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Add training for residual layers
"""
import os

import models.networks as networks
from lpips_pytorch import lpips, LPIPS
from util import util
from .EntropyBottleneck import EntropyBottleneck, TextureEntropyBottleneck_v3, TextureEntropyBottleneck, TextureEntropyBottleneck_v2, TextureEntropyBottleneck_v4, TextureEntropyBottleneck_v1,TextureEntropyBottleneck_v0
from .networks import *
from perceptual_similarity import perceptual_loss as ps

tensor_kwargs = {"dtype": torch.float32, "device": torch.device("cuda:0")}


class CSEANModel_2(torch.nn.Module):
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

        self.texture_entropy = TextureEntropyBottleneck(self.opt.style_nc, opt).to(**tensor_kwargs)
        self.texture_entropy.init_weights(opt.init_type, opt.init_variance)

        self.netE, self.netG, self.texture_entropy = self.initialize_base_networks(opt)
        self.set_requires_grad([self.netE, self.netG, self.texture_entropy], False)

        self.initialize_enhance_networks(opt)

        if opt.isTrain:
            self.loss_lpips = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(),gpu_ids=[self.opt.local_rank])
            # self.criterionGAN = networks.GANLoss(
            #     opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            # self.criterionL1 = torch.nn.L1Loss()
            self.loss_mse = torch.nn.MSELoss()
            # if self.opt.k_vgg:
            #     self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            init_loss = self.FloatTensor(1).fill_(0).cuda(self.opt.local_rank)
            self.loss = {"mse": init_loss, "lpips": init_loss, "bpp":init_loss}

    def recons_base(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)

        if torch.isnan(style_matrix).any():
            raise AssertionError("nan in style matrix")
        q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
            style_matrix / (2**self.opt.qp_step))
        q_style_matrix = q_style_matrix * (2**self.opt.qp_step)

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
#         print(texture_length.require_grad)
        decode_image = self.netG(input_semantics_onehot, q_style_matrix)

        return real_image, decode_image, tex_bpp

    def test(self,data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)

        if torch.isnan(style_matrix).any():
            raise AssertionError("nan in style matrix")
        q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
            style_matrix / (2 ** self.opt.qp_step))
        q_style_matrix = q_style_matrix * (2 ** self.opt.qp_step)

        decode_image = self.netG(input_semantics_onehot, q_style_matrix)
        latent_residual = self.residual_encoder(real_image, decode_image)
        latent_residual_hat, _, res_length = self.residual_entropy_model(latent_residual)
        decode_residual = self.residual_decoder(latent_residual_hat, decode_image)
        refine_img = self.refine_model(decode_residual, decode_image)

        return real_image,refine_img


    def residual_enhance(self,real_image,decode_image):
        latent_residual = self.residual_encoder(real_image, decode_image)
        latent_residual_hat, _, res_length = self.residual_entropy_model(latent_residual)
        decode_residual = self.residual_decoder(latent_residual_hat, decode_image)
        refine_img = self.refine_model(decode_residual, decode_image)
        res_bpp = res_length / (real_image.numel() / real_image.size(1))

        return refine_img,res_bpp

    def forward(self, data, mode='residual', regularize=False, style_code=None, lmbda=-1):
        loss = 0
        if mode == "val":
            decode_image, tex_bpp, q_style_matrix = self.val_forward(data)
            return decode_image, tex_bpp, q_style_matrix
        elif mode == "test_forward":
            real_image, decode_image, tex_bpp, q_style_matrix, noise_scale = self.test_forward(data)
            return real_image, decode_image, tex_bpp, q_style_matrix, noise_scale
        elif mode == "transfer":
            transfer_img = self.transfer_forward(data, style_code)
            return transfer_img
        elif mode == "encode":
            style_matrix = self.encode_forward(data)
            return style_matrix
        elif mode == "test":
            real_img, refine_img = self.test(data)
            return real_img,refine_img
        elif mode == "residual":
            real_image, decode_image, tex_bpp = self.recons_base(data)
            refine_img, res_bpp= self.residual_enhance(real_image,decode_image)
            self.refine_img = refine_img
            self.loss['bpp'] = res_bpp
            self.loss['mse'] = self.loss_mse((real_image+1.0)/2.0*255.0,(refine_img+1.0)/2.0*255.0)
            self.loss['lpips'] = torch.mean(self.loss_lpips.forward(real_image,refine_img))
            total_loss = self.opt.lmbda * self.loss['bpp'] + self.opt.k_mse * self.loss['mse'] + self.opt.k_lpips * self.loss['lpips']
            return total_loss, res_bpp, refine_img, self.loss
        else:
            return loss


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
        util.save_network(self.residual_entropy_model, 'residual_entropy', epoch, self.opt)
        util.save_network(self.residual_encoder, 'resE', epoch, self.opt)
        util.save_network(self.residual_decoder, 'resD', epoch, self.opt)
        util.save_network(self.refine_model, 'refine', epoch, self.opt)
        optimizer = {
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(optimizer, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pkl'))


    ############################################################################
    # Private helper methods
    ############################################################################
    def _param_num(self):
        semantic_codec_sum = sum(p.numel() for p in self.semantic_codec.parameters() if p.requires_grad)
        entroy_sum = sum(p.numel() for p in self.texture_entropy.parameters() if p.requires_grad)
        print("semantic codec trainable parameters: %d\n" % semantic_codec_sum)
        print("texture entropy model trainable parameters: %d\n" % entroy_sum)

    def initialize_base_networks(self, opt):
        print("********", opt.semantic_nc)
        netG = networks.define_G(opt)
        netE = networks.define_E(opt)
        self.texture_entropy = util.load_network(self.texture_entropy, 'texture_entropy', opt.base_epoch, opt)
        print("load texture entropy success")
        self.netG = util.load_network(netG, 'G', opt.base_epoch, opt)
        print("load netG success")
        # print(netE.named_parameters)
        self.netE = util.load_network(netE, 'E', opt.base_epoch, opt)
        print("load netE success")

        return self.netE, self.netG, self.texture_entropy

    def create_network(self,net,opt):
        net.print_network()
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.cuda()
        net.init_weights(opt.init_type, opt.init_variance)
        return net

    def initialize_enhance_networks(self, opt):
        self.residual_encoder = self.create_network(Residual_Encoder(6, 96, 16),opt)
        self.residual_decoder = self.create_network(Residual_Decoder(16, 96, 3),opt)
        self.refine_model = self.create_network(Refinement(3, 48, 64, 3),opt)
        self.residual_entropy_model = self.create_network(ResidualEntropyBottleneck(16, 96),opt)

        if not opt.isTrain or opt.continue_train:
            self.residual_entropy_model = util.load_network(self.residual_entropy_model, 'residual_entropy', opt.which_epoch, opt)
            print("load residual entropy success")
            self.residual_encoder = util.load_network(self.residual_encoder, 'resE', opt.which_epoch, opt)
            print("load residual_encoder success")
            # print(netE.named_parameters)
            self.residual_decoder = util.load_network(self.residual_decoder, 'resD', opt.which_epoch, opt)
            print("load residual_decoder success")
            self.refine_model = util.load_network(self.refine_model, 'refine', opt.which_epoch, opt)
            print("load refine_model success")

        if self.opt.isTrain:
            self.optimizer = self.create_optimizers(opt)

        if opt.resume:
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            optim_ckpt = torch.load(os.path.join(save_dir, 'optimizer.pkl'))
            self.optimizer.load_state_dict(optim_ckpt['optimizer'])


    def create_optimizers(self, opt):
        G_params = list(self.residual_encoder.parameters())
        G_params += list(self.residual_decoder.parameters())
        G_params += list(self.residual_entropy_model.parameters())
        G_params = list(self.refine_model.parameters())

        G_lr = opt.lr
        beta1, beta2 = opt.beta1, opt.beta2
        optimizer = torch.optim.Adam(G_params, lr=opt.lr, betas=(beta1, beta2))

        return optimizer


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

    def val_forward(self, data):
        input_semantic = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantic)
        style_matrix = self.netE(real_image, input_semantic).cuda(self.opt.local_rank)

        if self.opt.adap_qp:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix)
        elif self.opt.binary_quant:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix1 = q_style_matrix1 * (2 ** self.opt.qp_step)
        else:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix1 = q_style_matrix1 * self.opt.qp_step

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
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2 ** self.opt.qp_step)
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

    def test_vis(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        vis_feature = self.netE(real_image, input_semantics)  # (bs,512,19,1)
        # style_matrix,x_layer3 = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)
        return vis_feature

    def recons_prepare(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)

        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding

        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)
        if torch.isnan(style_matrix).any():
            raise AssertionError("nan in style matrix")
        if self.opt.binary_quant:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2 ** self.opt.qp_step)
        else:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix = q_style_matrix * self.opt.qp_step

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
        #         print(texture_length.require_grad)
        decode_image = self.netG(input_semantics_onehot, q_style_matrix)
        restyle_matrix = self.netE(decode_image, input_semantics).cuda(self.opt.local_rank)
        if self.opt.binary_quant:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / (2 ** self.opt.qp_step))
            q_restyle_matrix = q_restyle_matrix * (2 ** self.opt.qp_step)
        else:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / self.opt.qp_step)
            q_restyle_matrix = q_restyle_matrix * self.opt.qp_step

        return tex_bpp, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix

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
