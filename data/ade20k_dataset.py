"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

from data.image_folder import make_dataset
from data.pix2pix_dataset import Pix2pixDataset


class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        # parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []

        for item in os.listdir(os.path.join(root, "images")):
            img_dir = os.path.join(root, "images")
            img_dir = os.path.join(img_dir, item)
            label_dir = os.path.join(root, "labels")
            label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
            if not os.path.exists(label_dir):
                continue
            else:
                image_paths.append(img_dir)
                label_paths.append(label_dir)
        # for p in all_images:
        #     if '_%s_' % phase not in p:
        #         continue
        #     if p.endswith('.jpg'):
        #         image_paths.append(p)
        #     elif p.endswith('.png'):
        #         if opt.label_type == 'edge':
        #             if 'edge' in p:
        #                 label_paths.append(p)
        #         else:
        #             label_paths.append(p)

        instance_paths = []  # don't use instance map for ade20k

        return label_paths, image_paths, instance_paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
