"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.sean_model import SEANModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = SEANModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None  # 生成的图像
        if opt.isTrain:
            if opt.with_entropy:
                self.optimizer_G, self.optimizer_C = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            else:
                self.optimizer_G, self.optimizer_D = \
                    self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    # for codec version: semantic map codec + texture code codec
    def run_one_step(self, data):
        self.optimizer_G.zero_grad()
        loss, loss_sem, bpp, sem_bpp, decode_image, decode_semantic = self.pix2pix_model_on_one_gpu(data)
        loss = sum(loss.values()).mean()
        loss.backward()
        self.optimizer_G.step()

        self.optimizer_C.zero_grad()
        loss_sem = sum(loss_sem.values()).mean()
        loss_sem.backward()
        self.optimizer_C.step()

        self.loss,self.loss_sem=loss,loss_sem
        self.bpp,self.sem_bpp=bpp,sem_bpp
        self.decode_image,self.decode_semantic=decode_image,decode_semantic

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        # print("g_loss", g_losses)
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data, regularize=False):
        self.optimizer_D.zero_grad()
        if regularize:
            d_losses,r1_loss=self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            # r1_loss = sum()
            d_loss.backward()
            r1_loss.backward()
        else:
            d_losses = self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
        self.optimizer_D.step()
        # print("d_loss", d_loss)
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

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
