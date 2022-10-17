import os

from skimage.io import imread
from torch.utils.data import Dataset

from .base_dataset import *


class TrainDataBase(Dataset):
    def __init__(self, block_size):
        super(TrainDataBase, self).__init__()
        self._bs = block_size
        self._images = list()

        self._tf = transforms.Compose([
            transforms.RandomCrop(self._bs),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self._images)


class TrainPairedData(Dataset):
    def __init__(self, root, opt):
        super().__init__()
        self.opt = opt
        self.images = list()
        self.labels = list()
        if self.opt.use_celeba:
            self.load_dataset(root, "celeba")
        if self.opt.use_ffhq:
            self.load_dataset(root, "ffhq")
        if self.opt.use_id:
            self.load_dataset(root, "id")
        if len(self.images) < 1:
            for item in os.listdir(os.path.join(root, "images")):
                img_dir = os.path.join(root, "images")
                img_dir = os.path.join(img_dir, item)
                if self.opt.labelroot is not None:
                    label_dir = os.path.join(root, self.opt.labelroot)
                elif self.opt.labels_x8:
                    label_dir = os.path.join(root, "labels_x8")
                    if not os.path.exists(label_dir):
                        label_dir = os.path.join(root, "labels")
                else:
                    label_dir = os.path.join(root, "labels")
                # label_dir = os.path.join(label_dir, str((item.split('.')[0]).split('_')[-1]) + ".png")
#                 print(label_dir)
                if self.opt.dataset_mode == "cityscapes":
#                     print(item)
                    label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
                elif self.opt.dataset_mode == "ade20k":
                    label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
                else:
                    label_dir = os.path.join(label_dir, str(item.split('_')[-1].split('.')[0]) + ".png")
                # label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
               
                if not os.path.exists(label_dir):
                    continue
                else:
                    self.images.append(img_dir)
                    self.labels.append(label_dir)
#             print(root)
            print(len(self.images), len(self.labels))

    def load_dataset(self, root, name):
        if not os.path.exists(os.path.join(root, "images" + '_' + name)):
            return
        for item in os.listdir(os.path.join(root, "images" + '_' + name)):
            img_dir = os.path.join(root, "images" + '_' + name)
            img_dir = os.path.join(img_dir, item)
            label_dir = os.path.join(root, "labels" + '_' + name)
            # label_dir = os.path.join(label_dir, str((item.split('.')[0]).split('_')[-1]) + ".png")
            label_dir = os.path.join(label_dir, str(item.split('.')[0]) + ".png")
            if not os.path.exists(label_dir):
                continue
            else:
                self.images.append(img_dir)
                self.labels.append(label_dir)

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext or 'input_' + filename1_without_ext == filename2_without_ext or filename1_without_ext == 'input_' + filename2_without_ext

    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        assert self.paths_match(image_path, label_path), "The label_path %s and image_path %s don't match." % \
                                                         (label_path, image_path)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # import pdb
        # pdb.set_trace()

        params = get_params(self.opt, image.size)
        transforms_image = get_transform(self.opt, params, method=Image.LANCZOS)
        transforms_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        image_tensor = transforms_image(image)
        label_tensor = transforms_label(label)
        label_tensor = label_tensor * 255

        # label_tensor[label_tensor == 255] = self.opt.label_nc

        input_dict = {'image': image_tensor,
                      'label': label_tensor,
                      'path': image_path,
                      }

        if self.opt.dataset_mode == "ade20k":
            self.postprocess(input_dict)

        return input_dict

    def __len__(self):
        return len(self.images)


class TrainData(TrainDataBase):
    def __init__(self, block_size, root):
        super(TrainData, self).__init__(block_size)

        #         root = get_addr()
        for item in os.listdir(root):
            file = os.path.join(root, item)
            image = imread(file)
            self._images.append(image)

    def __getitem__(self, index):
        image = self._images[index]
        image = Image.fromarray(image)
        # print(image.size)
        if image.size[0] < 256 or image.size[1] < 256:
            image = image.resize((300, 300), Image.NEAREST)
        block = self._tf(image)

        return block


class TrainData2(TrainDataBase):
    def __init__(self, block_size, root):
        super(TrainData2, self).__init__(block_size)

        #         root = get_addr()
        for item in os.listdir(root):
            file = os.path.join(root, item)
            self._images.append(file)

    @staticmethod
    def pil_loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        file = self._images[index]
        image = self.pil_loader(file)
        block = self._tf(image)

        return block


class TestData(Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.images = list()

        for item in os.listdir(dataroot):
            file = os.path.join(dataroot, item)
            image = imread(file)
            self.images.append(image)

        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        index = index % len(self.images)
        image = self.images[index]
        image = Image.fromarray(image)
        # image = image.resize((256, 256), Image.LANCZOS)
        # if image.size[0] < 256 or image.size[1] < 256:
        #     image = image.resize((300, 300), Image.LANCZOS)
        image = self.transforms(image)

        return image
