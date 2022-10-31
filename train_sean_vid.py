"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.distributed as dist
from tqdm import tqdm
import gc

from lpips_pytorch import lpips
from data.DataRead import *
from options.train_options import TrainOptions
from trainers.end2end_trainer import End2EndTrainer
from util import *
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))


class TrainBase(object):
    def __init__(self, opt):
        super(TrainBase, self).__init__()
        self.opt = opt
        if opt.distributed:
            torch.cuda.set_device(opt.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()
        self._init_loaders()
        # create trainer for our model
        self.trainer = End2EndTrainer(opt)
        self.trainer.model_on_one_gpu.train()
        # create tool for counting iterations
        self.iter_counter = IterationCounter(opt, len(self.train_loader))
        # create tool for visualization
        self.visualizer = Visualizer(opt)
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.anchor = LineFit()

    def data_sampler(self, dataset, shuffle, distributed):
        if distributed:
            return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

        if shuffle:
            return data.RandomSampler(dataset)

        else:
            return data.SequentialSampler(dataset)

    def _init_loaders(self):
        # train_dataset = TrainPairedData(self.opt.train_dataroot, self.opt)
        train_dataset = TrainVidData(self.opt.train_img_csv,self.opt.train_lab_csv,self.opt)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batchSize,
            sampler=self.data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )


    def _setting(self):
        with open(os.path.join(self.log_dir, "setting.txt"), 'w') as f:
            namespace = vars(self.opt)
            for key in namespace:
                f.write("{}:\t {}\n".format(key, namespace[key]))

    def _print_infor(self, **kwargs):
        raise NotImplementedError

    def _save_imgae(self, data_i, epoch, step):
        # if self.opt.swap_train:
        input1 = ((data_i['P_image'][0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
        input1 = input1.detach().cpu().float().numpy()
        label1 = (data_i['P_label'][0]).round_().detach().cpu().float().numpy()
        if self.opt.swap_train:
            output1 = ((self.trainer.decode_image[0][0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
        else:
            output1 = ((self.trainer.decode_image[0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
       
        output1 = output1.detach().cpu().float().numpy()
        input1 = np.transpose(input1, (1, 2, 0))
        label1 = np.transpose(label1, (1, 2, 0))
        output1 = np.transpose(output1, (1, 2, 0))
        img_path1 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
            epoch, self.iter_counter.total_steps_so_far, "input1", step)
        label_path1 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
            epoch, self.iter_counter.total_steps_so_far, "label1", step)
        output_path1 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
            epoch, self.iter_counter.total_steps_so_far, "decode1", step)

        dir_path = os.path.join(self.opt.checkpoints_dir, self.opt.name + "/image/")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        Image.fromarray(np.uint8(input1)).save(os.path.join(dir_path, img_path1))
        print(os.path.join(dir_path, img_path1))
        Image.fromarray(np.uint8(np.squeeze(label1)), mode='L').save(os.path.join(dir_path, label_path1))
        print(os.path.join(dir_path, label_path1))
        Image.fromarray(np.uint8(output1)).save(os.path.join(dir_path, output_path1))
        print(os.path.join(dir_path, output_path1))

        if self.opt.swap_train:
            input2 = ((data_i['image'][int(self.opt.batchSize / 2)] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
            input2 = input2.detach().cpu().float().numpy()
            label2 = (data_i['label'][int(self.opt.batchSize / 2)]).round_().detach().cpu().float().numpy()

            output2 = ((self.trainer.decode_image[1][0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
            output2 = output2.detach().cpu().float().numpy()

            input2 = np.transpose(input2, (1, 2, 0))
            label2 = np.transpose(label2, (1, 2, 0))
            output2 = np.transpose(output2, (1, 2, 0))

            img_path2 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                epoch, self.iter_counter.total_steps_so_far, "input2", step)
            label_path2 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                epoch, self.iter_counter.total_steps_so_far, "label2", step)
            output_path2 = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                epoch, self.iter_counter.total_steps_so_far, "decode2", step)

            Image.fromarray(np.uint8(input2)).save(os.path.join(dir_path, img_path2))
            # print(os.path.join(dir_path, img_path1))
            Image.fromarray(np.uint8(np.squeeze(label2)), mode='L').save(os.path.join(dir_path, label_path2))
            # print(os.path.join(dir_path, label_path1))
            Image.fromarray(np.uint8(output2)).save(os.path.join(dir_path, output_path2))
            # print(os.path.join(dir_path, output_path1))

    def train(self):
        self._setting()
        self.iter = -1
        best_s = 100
        if self.opt.k_cd:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.trainer.optimizer_CD, max_lr=0.0002,
                                                            steps_per_epoch=len(self.train_loader), epochs=3)
        for epoch in self.iter_counter.training_epochs():
            # self.iter_counter.record_epoch_start(epoch)
            if self.iter > self.opt.total_step:
                break
            for i, data_i in enumerate(self.train_loader):
                self.iter = self.iter + 1
                if self.iter > self.opt.total_step:
                    break
                # Training
                self.trainer.run_one_step(data_i)
                torch.cuda.empty_cache()
                if self.opt.swap_train and i % self.opt.d_reg_every:
                    self.trainer.run_discriminator_one_step(data_i, True)
                elif not self.opt.fixed_EGD and not self.opt.fix_D:
                    self.trainer.run_discriminator_one_step(data_i)
                    torch.cuda.empty_cache()
                # print(self.iter_counter.total_steps_so_far)
                # Visualizations
                if self.iter % 500 == 0 and self.iter:
                    if self.opt.train_semantic:
                        results = self._print_infor(iteration=self.iter,
                                                    mse=self.trainer.loss_item[1].item(),
                                                    lpips=self.trainer.loss_item[0].item() * self.opt.k_lpips,
                                                    sem_mse=self.trainer.sem_mse.item() * self.opt.k_sem,
                                                    bpp=self.trainer.bpp.item(),
                                                    # tex_bpp=self.trainer.tex_bpp.item(),
                                                    sem_bpp=self.trainer.sem_bpp.item(),
                                                    loss=self.trainer.loss.item(),
                                                    )
                    else:
                        results = self._print_infor(iteration=self.iter,
                                                    mse=torch.mean(self.trainer.loss_item["mse"]).item(),
                                                    L1=torch.mean(self.trainer.loss_item["L1"]).item(),
                                                    lpips=torch.mean(self.trainer.loss_item["lpips"]).item(),
                                                    gan_feat=torch.mean(self.trainer.loss_item["GAN_Feat"]).item(),
                                                    gan=torch.mean(self.trainer.loss_item["GAN"]).item(),
                                                    g_coocur_loss=torch.mean(
                                                        self.trainer.loss_item["g_coocur_loss"]).item(),
                                                    latent=torch.mean(self.trainer.loss_item["latent"]).item(),
                                                    latent2=torch.mean(self.trainer.loss_item["latent2"]).item(),
                                                    latent_reg=torch.mean(self.trainer.loss_item["latent_reg"]).item(),
                                                    latent_ratio=torch.mean(self.trainer.loss_item["latent_ratio"]).item(),
                                                    latent_reg2=torch.mean(
                                                        self.trainer.loss_item["latent_reg2"]).item(),
                                                    latent_reg2_l2=torch.mean(
                                                        self.trainer.loss_item["latent_reg2_l2"]).item(),
                                                    recon_reg=torch.mean(self.trainer.loss_item["recon_reg"]).item(),
                                                    vgg=torch.mean(self.trainer.loss_item["VGG"]).item(),
                                                    d_fake=torch.mean(self.trainer.d_loss_item["D_fake"]).item(),
                                                    d_real=torch.mean(self.trainer.d_loss_item["D_real"]).item(),
                                                    d_coocur_fake=torch.mean(
                                                        self.trainer.d_loss_item["D_occur_fake"]).item(),
                                                    d_coocur_real=torch.mean(
                                                        self.trainer.d_loss_item["D_occur_real"]).item(),
                                                    bpp=torch.mean(self.trainer.bpp).item(),
                                                    # tex_bpp=torch.mean(self.trainer.tex_bpp).item(),
                                                    loss=torch.mean(self.trainer.loss).item(),
                                                    )

                    with open(os.path.join(self.log_dir, "log.txt"), 'a') as f:
                        f.write("epoch: {}\t iter:{}\t ".format(epoch, self.iter) + results + '\n')
                    if self.opt.local_rank in [-1, 0]:
                        self.trainer.save("latest")
                    # self.iter_counter.record_current_iter()
                    torch.cuda.empty_cache()
                    avg_lpips = self.test()
                    if avg_lpips < best_s:
                        best_s = avg_lpips
                        if self.opt.local_rank in [-1, 0]:
                            self.trainer.save('best')
                        print("best model with vgg score %.4f updated\n" % best_s)
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, self.iter))
                    print("[Iter {:<5d}] ".format(self.iter), end='\t')
                    print(results, end=' \t')

                if self.iter % 500 == 0 and self.iter:
                    # import pdb
                    # pdb.set_trace()
                    self._save_imgae(data_i, epoch, i)
                    torch.cuda.empty_cache()
                if self.opt.k_cd:
                    scheduler.step()
            self.trainer.update_learning_rate(epoch)
            # self.iter_counter.record_epoch_end()
            torch.cuda.empty_cache()
            print('saving the model of epoch %d' % epoch)
            if self.opt.local_rank in [-1, 0]:
                self.trainer.save(epoch)
        self.writer.flush()
        self.writer.close()
        print('Training was successfully finished.')


class WithoutTest(TrainBase):
    def __init__(self, *args, **kwargs):
        super(WithoutTest, self).__init__(*args, **kwargs)
        torch.backends.cudnn.benchmark = True

    def _print_infor(self, **kwargs):
        lr = min(param_group["lr"] for param_group in self.trainer.optimizer_G.param_groups)
        if self.opt.k_mse:
            psnr = 10 * np.log10(255 ** 2 / (kwargs["mse"]))
        else:
            psnr = 0.0

        self.writer.add_scalar("tex_bpp", kwargs["bpp"], global_step=kwargs["iteration"])
        self.writer.add_scalar("Training Loss", kwargs["loss"], global_step=kwargs["iteration"])
        self.writer.add_scalar('PSNR', psnr, global_step=kwargs["iteration"])
        self.writer.add_scalar("lpips", kwargs["lpips"], global_step=kwargs["iteration"])
        self.writer.add_scalar("gan_feat", kwargs["gan_feat"], global_step=kwargs["iteration"])
        self.writer.add_scalar("latent", kwargs["latent"], global_step=kwargs["iteration"])
        self.writer.add_scalars("GAN", {"GAN_loss": kwargs["gan"], "D_loss_fake": kwargs["d_fake"],
                                        "D_loss_real": kwargs["d_real"]},
                                global_step=kwargs["iteration"])

        if self.opt.train_semantic:
            return "LR: {:>3.1e}\t Loss: {:>.4f}\t PSNR: {:>5.2f} dB\t sem_bpp:{:>.4f}\t tex_bpp:{:>.4f}\t lpips: {:>.4f}\t image_mse {:>.4f}\t sem_mse: {:>.4f}\n".format(
                lr, kwargs["loss"], psnr, kwargs["sem_bpp"], kwargs["bpp"], kwargs["lpips"], kwargs["mse"],
                kwargs["sem_mse"]
            )
        else:
            return "LR: {:>3.1e}\t Loss: {:>.4f}\t PSNR: {:>5.2f} dB\t tex_bpp:{:>.4f}\t weighted_tex_bpp:{:>.4f}\t  weighted_L1: {:>.4f}\t weighted_mse {:>.4f}\t weighted_gan_feat {:>.4f}\t" \
                   " weighted_latent_reg {:>.4f}\t weighted_recon_reg {:>.4f}\t latent_reg_with_real {:>.4f}\t latent_reg_real_L2 {:>.4f}\t latent_ratio {:>.4f}\t weighted_vgg_loss {:>.4f}\t weigted_latent_loss {:>.4f}\t weigted_latent2_loss {:>.4f}\t" \
                   " weighted_g_coocur {:>.4f}\t weighted_gan {:>.4f}\t  d_fake {:>.4f}\t d_real {:>.4f}\t d_coocur_fake {:>.4f}\t d_coocur_real {:>.4f}\t".format(
                lr, kwargs["loss"], psnr, kwargs["bpp"],kwargs["bpp"]*self.opt.lmbda, kwargs["L1"] * self.opt.k_L1,
                                                         kwargs["mse"] * self.opt.k_mse,
                                                         self.opt.k_feat * kwargs['gan_feat'],
                                                         self.opt.k_latent_reg * kwargs['latent_reg'],
                                                         self.opt.k_recon_reg * kwargs['recon_reg'],
                                                         self.opt.k_latent_reg2 * kwargs['latent_reg2'],
                                                         self.opt.k_latent_reg2_l2 * kwargs['latent_reg2_l2'],
                                                         self.opt.k_latent_ratio * kwargs['latent_ratio'],
                                                         self.opt.k_vgg * kwargs['vgg'],
                                                         kwargs['latent'] * self.opt.k_latent,
                                                         kwargs['latent2'] * self.opt.k_latent2,
                                                         kwargs['g_coocur_loss'] * self.opt.k_cd,
                                                         kwargs["gan"] * self.opt.k_gan, kwargs['d_fake'],
                kwargs['d_real'], kwargs['d_coocur_fake'], kwargs['d_coocur_real'])


class WithTest(TrainBase):
    def _init_loaders(self):
        super(WithTest, self)._init_loaders()
        test_dataset = TrainVidData(self.opt.test_img_csv,self.opt.test_lab_csv, self.opt)
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batchSize,
            sampler=self.data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def test(self):
        lpips_list = list()
        self.trainer.model_on_one_gpu.eval()
        with torch.no_grad():
            for inputs in self.test_loader:
                real_image, decode_image = self.trainer.model_on_one_gpu(
                    inputs, mode="test_vid")
                # psnr = 10 * torch.log10(255 ** 2 / loss_item[1])
                lpips_score = lpips(self.opt, real_image, decode_image)
                lpips_list.append(lpips_score.item())

        lpips_score = np.mean(lpips_list)
        return lpips_score

    def _print_infor(self, **kwargs):
        # results = self.test()
        lr = min(param_group["lr"] for param_group in self.trainer.optimizer_G.param_groups)
        if self.opt.k_mse:
            psnr = 10 * np.log10(255 ** 2 / (kwargs["mse"]))
        else:
            psnr = 0.0
        self.writer.add_scalar("tex_bpp", kwargs["bpp"], global_step=kwargs["iteration"])
        self.writer.add_scalar("Training Loss", kwargs["loss"], global_step=kwargs["iteration"])
        self.writer.add_scalar('PSNR', psnr, global_step=kwargs["iteration"])
        self.writer.add_scalar("lpips", kwargs["lpips"], global_step=kwargs["iteration"])
        self.writer.add_scalar("gan_feat", kwargs["gan_feat"], global_step=kwargs["iteration"])
        self.writer.add_scalar("latent", kwargs["latent"], global_step=kwargs["iteration"])
        self.writer.add_scalars("GAN", {"GAN_loss": kwargs["gan"], "D_loss_fake": kwargs["d_fake"],
                                        "D_loss_real": kwargs["d_real"]},
                                global_step=kwargs["iteration"])

        if self.opt.train_semantic:
            return "LR: {:>3.1e}\t Loss: {:>.4f}\t PSNR: {:>5.2f} dB\t sem_bpp:{:>.4f}\t tex_bpp:{:>.4f}\t lpips: {:>.4f}\t image_mse {:>.4f}\t sem_mse: {:>.4f}\n".format(
                lr, kwargs["loss"], psnr, kwargs["sem_bpp"], kwargs["bpp"], kwargs["lpips"], kwargs["mse"],
                kwargs["sem_mse"]
            )
        else:
            return "LR: {:>3.1e}\t Loss: {:>.4f}\t PSNR: {:>5.2f} dB\t tex_bpp:{:>.4f}\t weighted_tex_bpp:{:>.4f}\t  weighted_L1: {:>.4f}\t weighted_mse {:>.4f}\t weighted_gan_feat {:>.4f}\t" \
                   " weighted_latent_reg {:>.4f}\t weighted_recon_reg {:>.4f}\t latent_reg_with_real {:>.4f}\t latent_reg_real_L2 {:>.4f}\t latent_ratio {:>.4f}\t weighted_vgg_loss {:>.4f}\t weigted_latent_loss {:>.4f}\t weigted_latent2_loss {:>.4f}\t" \
                   " weighted_g_coocur {:>.4f}\t weighted_gan {:>.4f}\t  d_fake {:>.4f}\t d_real {:>.4f}\t d_coocur_fake {:>.4f}\t d_coocur_real {:>.4f}\t".format(
                lr, kwargs["loss"], psnr, kwargs["bpp"],kwargs["bpp"]*self.opt.lmbda, kwargs["L1"] * self.opt.k_L1,
                                                         kwargs["mse"] * self.opt.k_mse,
                                                         self.opt.k_feat * kwargs['gan_feat'],
                                                         self.opt.k_latent_reg * kwargs['latent_reg'],
                                                         self.opt.k_recon_reg * kwargs['recon_reg'],
                                                         self.opt.k_latent_reg2 * kwargs['latent_reg2'],
                                                         self.opt.k_latent_reg2_l2 * kwargs['latent_reg2_l2'],
                                                         self.opt.k_latent_ratio * kwargs['latent_ratio'],
                                                         self.opt.k_vgg * kwargs['vgg'],
                                                         kwargs['latent'] * self.opt.k_latent,
                                                         kwargs['latent2'] * self.opt.k_latent2,
                                                         kwargs['g_coocur_loss'] * self.opt.k_cd,
                                                         kwargs["gan"] * self.opt.k_gan, kwargs['d_fake'],
                kwargs['d_real'], kwargs['d_coocur_fake'], kwargs['d_coocur_real'])


if __name__ == "__main__":

    if opt.with_test:
        obj = WithTest(opt)
    else:
        obj = WithoutTest(opt)
    obj.train()
