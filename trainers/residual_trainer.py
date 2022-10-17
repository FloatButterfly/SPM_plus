# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
# import torchprof
from torch import nn
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.sean_compression_model import CSEANModel
from models.sean_model_v2 import CSEANModel_2


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
        self.model = CSEANModel_2(opt).cuda(opt.local_rank)
#         print(self.model)
        self.loss_item = None

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
            self.optimizer = self.model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    # for codec version: semantic map codec + texture code codec
    def run_one_step(self, data, lmbda=-1):
        # with torch.autograd.set_detect_anomaly(True):
        self.optimizer.zero_grad()
        self.loss, self.bpp, self.decode_image, self.loss_item = self.model_on_one_gpu(data, mode="residual",
                                                                                               lmbda=lmbda)
        self.loss.backward()
        self.optimizer.step()




    def get_latest_losses(self):
        return {**self.loss, **self.loss_item}

    def get_latest_generated(self):
        return {**self.decode_image, **self.decode_image}

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

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


    def initiate_learning_rate(self, opt):
        new_lr = opt.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('initiate learning rate for new rate: %f' % (new_lr))
        self.old_lr=opt.lr

