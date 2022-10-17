# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torchprof
from torch import nn
from apex import amp
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.sean_compression_model import CSEANModel
from models.sean_model_v1 import CSEANModel_1


class End2EndTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        # if opt.ablation:
        #     self.model=ablation.CSEANModel(opt)
        self.model = CSEANModel_1(opt).cuda(opt.local_rank)
#         print(self.model)
        self.loss_item = None
        if opt.isTrain:
            self.d_loss_item = self.model.D_losses
        if len(opt.gpu_ids) > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
            )

            # self.model = DataParallelWithCallback(self.model,
            #                                       device_ids=opt.gpu_ids)
            self.model_on_one_gpu = self.model.module
            self.model_on_one_gpu = self.model_on_one_gpu.cuda()
        else:
            self.model_on_one_gpu = self.model
            torch.cuda.set_device(0)

        self.decode_image = None  # 生成的图像
        if opt.isTrain:
            if self.opt.k_cd:
                self.optimizer_G, self.optimizer_D, self.optimizer_CD = \
                    self.model_on_one_gpu.create_optimizers(opt)
            else:
                self.optimizer_G, self.optimizer_D = self.model_on_one_gpu.optimizer_G, self.model_on_one_gpu.optimizer_D
            self.old_lr = opt.lr

    # for codec version: semantic map codec + texture code codec
    def run_one_step(self, data, lmbda=-1):
        # with torch.autograd.set_detect_anomaly(True):
        self.optimizer_G.zero_grad()
        # with torchprof.Profile(self.model_on_one_gpu, use_cuda=True) as prof:

        if self.opt.swap_train:
            self.loss, self.bpp, decode_image1, decode_image2, self.loss_item = self.model_on_one_gpu(data,
                                                                                                      mode="generator")
            self.decode_image = [decode_image1, decode_image2]
        # print(prof.display(show_events=False))
        else:
            self.loss, self.bpp, self.decode_image, self.loss_item = self.model_on_one_gpu(data, mode="generator",
                                                                                               lmbda=lmbda)
        if self.opt.apex:
            with amp.scale_loss(self.loss,self.optimizer_G):
                    self.loss.backward()
        else:
            self.loss.backward()
#         for name, parms in self.model_on_one_gpu.named_parameters():
# #             if name.find('texture_entropy') != -1:
# #                 print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
# #                       ' -->grad_value:',parms.grad)
#             if name.find('netE') != -1:
#                 print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
#                       ' -->grad_value:',parms.grad)
#         print(self.loss_item['bpp'].grad)
#         print(self.loss,self.loss.require_grad)
#         print(self.loss.grad)
        # nn.utils.clip_grad_norm_(self.model_on_one_gpu.parameters(), 10)
        self.optimizer_G.step()
        # for name, params in self.model_on_one_gpu.named_parameters():
        #     print(name, params.requires_grad, params.grad.shape)
        # print(loss)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        # print("g_loss", g_losses)
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    # def run_discriminator_one_step(self, data):
    #     self.optimizer_D.zero_grad()
    #     d_losses = self.model_on_one_gpu(data, mode='discriminator')
    #     # d_loss = sum(d_losses.values()).mean()
    #     d_losses.backward()
    #     self.optimizer_D.step()
    #     # print("d_loss", d_loss)
    #     self.d_losses = d_losses

    def run_discriminator_one_step(self, data, regularize=False):
        self.optimizer_D.zero_grad()

        if self.opt.swap_train:
            if self.opt.k_cd:
                self.optimizer_CD.zero_grad()
            self.d_losses, self.d_loss_item = self.model_on_one_gpu(data, mode='discriminator', regularize=regularize)
            self.d_losses.backward()
            if self.opt.k_cd:
                self.optimizer_CD.step()
        else:
            self.d_losses, self.d_loss_item = self.model_on_one_gpu(data, mode='discriminator')
            if self.opt.apex:
                with amp.scale_loss(self.d_losses,self.optimizer_D):
                    self.d_losses.backward()
            else:
                self.d_losses.backward()

        # nn.utils.clip_grad_norm_(self.model_on_one_gpu.parameters(), 10)
        self.optimizer_D.step()

    def get_latest_losses(self):
        return {**self.loss, **self.loss_item}

    def get_latest_generated(self):
        return {**self.decode_image, **self.decode_semantic}

    # def update_learning_rate(self, epoch):
    #     self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr


    def initiate_learning_rate(self, opt):
        if self.opt.no_TTUR:
            new_lr_G = opt.lr
            new_lr_D = opt.lr
        else:
            new_lr_G = opt.lr / 2
            new_lr_D = opt.lr * 2

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('initiate learning rate for new rate: %f -> %f' % (new_lr_G, new_lr_D))
        self.old_lr=opt.lr

