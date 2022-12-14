"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os
import pickle
import sys

import torch

import data
import models
from util import util


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='sean_codec/celeba/',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the first epoch to start')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop',
                            help='scaling and cropping of images at load time.', choices=(
                "resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside",
                "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=286,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=18,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label. 3 if edge')  # coco 182; ADE20K 150
        parser.add_argument('--labelroot', type=str, default=None)
        parser.add_argument('--contain_dontcare_label', action='store_true', default=False,
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--style_nc', type=int, default=512, help="the channels of style matrix")
        parser.add_argument('--label_type', type=str, default='semantic', help='type of label: edge|semantic|')
        # for setting inputs
        parser.add_argument('--use_celeba', action='store_true')
        parser.add_argument('--use_ffhq', action='store_true')
        parser.add_argument('--use_id', action='store_true')
        parser.add_argument('--train_dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--test_dataroot', type=str, default='./datasets/val')
        parser.add_argument('--dataroot', type=str, default='./datasets/val')
        parser.add_argument('--dataset_mode', type=str, default='coco')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true',
                            help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='style',
                            help='selects model to use for netG (pix2pixhd | spade | style | Unet)')
        parser.add_argument('--netE', type=str, default='style', help='model name for encoder: Resnet | style | Conv')
        # parser.add_argument('--netCD', type=str, default='Coo', help='model name for encoder: Resnet | style | Conv')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=32,
                            help="dimension of the latent z vector")

        # for contrastive discriminator:coor_channel,loadCD,cooccur_r1
        parser.add_argument('--coor_channel', type=int, default=32,
                            help="the input channel of constrastive discriminator")
        parser.add_argument("--channel_multiplier", type=int, default=1)
        parser.add_argument('--loadCD', action='store_true',
                            help='whether load the pretrained codec for contrastive discriminator ')
        parser.add_argument('--cooccur_r1', type=float, default=1, help='weight for contrastive discriminator loss')
        parser.add_argument('--d_reg_every', type=float, default=16, help='weight for contrastive discriminator loss')
        parser.add_argument("--ref_crop", type=int, default=4)
        parser.add_argument("--n_crop", type=int, default=8)

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true',
                            help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

        # for rate control codec
        parser.add_argument('--codec_nc', type=int, default=64, help="channels of codes in end-to-end codec")
        parser.add_argument('--load_codec', action='store_true', help='whether load the pretrained codec for semantic ')
        parser.add_argument('--k_lpips', type=float, default=0, help='weight of lpips distortion')
        parser.add_argument('--k_mse', type=float, default=0, help='weight of mse')
        parser.add_argument('--k_sem', type=float, default=0, help='weight of semantic mse')
        parser.add_argument('--k_feat', type=float, default=1.0, help='weight for feature matching loss')
        parser.add_argument('--k_vgg', type=float, default=1.0, help='weight for vgg loss')
        parser.add_argument('--k_gan', type=float, default=1.0, help='weight for hinge gan loss')
        parser.add_argument('--k_latent', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--k_latent2', type=float, default=0,
                            help='weight for latent regression loss of texture codes between hybrid image and source image')
        parser.add_argument('--k_cd', type=float, default=0.0, help='weight for contrastive discriminator loss')
        parser.add_argument('--k_L1', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--k_latent_reg', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--k_recon_reg', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--k_latent_ratio', type=float, default=0, help='L2 for ratio distance loss')
        parser.add_argument('--k_latent_reg2', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--k_latent_reg2_l2', type=float, default=0.0, help='weight for latent regression loss')
        parser.add_argument('--tar_ratio', type=float, default=1.0, help='consis_dominator')
        parser.add_argument('--cont', type=float, default=1.0, help='consis_dominator')
        parser.add_argument('--qp_step', type=float, default=0.01,
                            help='quantization scale for style_matrix: 1e-2, 1e-3, 1e-4')

        # if train with entropy optimization
        parser.add_argument('--with_entropy', action="store_true", help='if train the code with entropy estimation')
        # parser.add_argument('--lmbda', type=float, default=10, help="rate param")
        parser.add_argument('--GANmode', action='store_true', help='whether use GAN loss.')
        parser.add_argument('--train_semantic', action='store_true', default=False,
                            help="whether joint training semantic map")
        parser.add_argument('--swap_train', action='store_true', help='whether use GAN loss.')
        parser.add_argument('--non_local', action='store_true', help='whether use non-local block in encoder.')
        parser.add_argument('--ds_label', action='store_true', help='whether use non-local block in encoder.')
        parser.add_argument('--ds_label_2', action='store_true', help='whether use non-local block in encoder.')
        parser.add_argument('--ds_label_4', action='store_true', help='whether use non-local block in encoder.')
        parser.add_argument('--ds_label_6', action='store_true', help='whether use non-local block in encoder.')
        parser.add_argument('--adap_qp', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--adap_qp_region_wise', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--train_continue_rate', action='store_true', help='whether training continue rate.')
        parser.add_argument('--fixed_EGD', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--fix_E', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--fix_G', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--fix_P', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--fix_D', action='store_true', help='whether use adaptive qp step.')
        parser.add_argument('--semantic_prior', action='store_true', help='whether to learn the cross semantic region prior.')
        parser.add_argument('--semantic_prior_v2', action='store_true',
                            help='whether to use texture entropy bottleneck v2.')
        parser.add_argument('--semantic_prior_v3', action='store_true',
                            help='whether to use texture entropy bottleneck v3.')
        parser.add_argument('--semantic_prior_v4', action='store_true',
                            help='whether to use texture entropy bottleneck v4.')

        parser.add_argument('--upper_bound', type=float, default=0.1, help="rate param")
        parser.add_argument('--lower_bound', type=float, default=0.01, help="rate param")
        parser.add_argument('--lmbda', type=float, default=10, help="rate param")
        # for distributed training
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument('--resume', action='store_true', help='Load optimizer state dict')
#         parser.add_argument('--non_local', action="store_true", help='if train the code with entropy estimation')
        parser.add_argument('--apex', action='store_true', help='where use mix fp16 to save memory')
        parser.add_argument('--labels_x8', action='store_true', help='Load labels_x8')
        parser.add_argument('--binary_quant', action='store_true', help='binary quantization, qp 2,4,6,8,10')
        parser.add_argument('--GSM', action='store_true', help='use GSM entropy model')
        parser.add_argument('--no_hyper', action='store_true', help='use full factorized model')
        self.initialized = True
        self.isTrain = False
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()

        n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        opt.distributed = n_gpu > 1

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
                          (1 if opt.contain_dontcare_label else 0) + \
                          (0 if opt.no_instance else 1)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
