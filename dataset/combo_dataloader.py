import os
import os.path
import random

from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import torch.utils.data as data
from dataset import pil_aug_transforms as pil_aug_trans
from dataset import cv2_aug_transforms as cv2_aug_trans
# from dataset import transforms as trans
import json
import logging
from .label_relax_transforms import RelaxedBoundaryLossToTensor

class Configer(object):
    def __init__(self, hypes_file=None):
        if hypes_file is not None:
            if not os.path.exists(hypes_file):
                logging.error('Json Path:{} not exists!'.format(hypes_file))
                exit(0)

            json_stream = open(hypes_file, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

    def exists(self, *key):
        if len(key) == 1 and key[0] in self.params_root:
            return True

        if len(key) == 2 and (key[0] in self.params_root and key[1] in self.params_root[key[0]]):
            return True
        return False
    
    def get(self, *key):
        if len(key) == 0:
            return self.params_root

        elif len(key) == 1:
            if key[0] in self.params_root:
                return self.params_root[key[0]]
            else:
                logging.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        elif len(key) == 2:
            if key[0] in self.params_root and key[1] in self.params_root[key[0]]:
                return self.params_root[key[0]][key[1]]
            else:
                logging.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        else:
            logging.error('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)



# ###### Data loading #######
def make_dataset(root, lst):
    # append all index
    fid = open(lst, 'r')
    imgs, segs = [], []
    for line in fid.readlines():
        idx = line.strip().split(' ')[0]
        image_path = os.path.join(root, 'JPEGImages/' + str(idx) + '.jpg')
        # image_path = os.path.join(root, str(idx) + '.jpg')
        seg_path = os.path.join(root, 'SegmentationPart/' + str(idx) + '.png')
        # seg_path = os.path.join(root, str(idx) + '.jpg')
        imgs.append(image_path)
        segs.append(seg_path)
    return imgs, segs


# ###### val resize & crop ######
def scale_crop(img, seg, crop_size):
    oh, ow = seg.shape
    pad_h = max(crop_size - oh, 0)
    pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
    pad_w = max(crop_size - ow, 0)
    pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                     value=(255,))
    else:
        img_pad, seg_pad = img, seg

    return img_pad, seg_pad


class DataGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.label_relax = RelaxedBoundaryLossToTensor(ignore_id=255, num_classes=7)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training
        self.configer = Configer(hypes_file='./dataset/data_augmentation_trans_config.json')

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            logging.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        # self.img_transform = trans.Compose([
        #     trans.ToTensor(),
        #     trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
        #                     mean=self.configer.get('normalize', 'mean'),
        #                     std=self.configer.get('normalize', 'std')), ])

        # self.label_transform = trans.Compose([
        #     trans.ToLabel(),
        #     trans.ReLabel(255, -1), ])

    def __getitem__(self, index):
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # load data
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            #random blur
            # gaussian blur as in PSP
            # if random.random() < 0.5:
            #     sigma = random.random()*10
            #     img = cv2.GaussianBlur(img, (int(sigma)*2+1,int(sigma)*2+1), int(sigma)+1)
            # gaussian blur as in PSP
            pil_img = Image.fromarray(img)
            if random.random() < 0.5:
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))
            img = np.asarray(pil_img)

            if self.aug_train_transform is not None:
                img, seg = self.aug_train_transform(img, labelmap=seg)

            # if self.img_transform is not None:
            #     img = self.img_transform(img)
            #
            # if self.label_transform is not None:
            #     seg = self.label_transform(seg)

            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape[:2]
            pad_h = max(self.crop_size - img_h, 0)
            pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
            pad_w = max(self.crop_size - img_w, 0)
            pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            seg_pad_h, seg_pad_w = seg_pad.shape
            h_off = random.randint(0, seg_pad_h - self.crop_size)
            w_off = random.randint(0, seg_pad_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]

            # Generate target maps
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        #label_relaxation
        lr_segmentations = self.label_relax(segmentations)

        return images, segmentations, segmentations_half, segmentations_full, lr_segmentations, name

    def __len__(self):
        return len(self.imgs)



class ub_DataGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training

    def __getitem__(self, index):
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # load data
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape[:2]
            pad_h = max(self.crop_size - img_h, 0)
            pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
            pad_w = max(self.crop_size - img_w, 0)
            pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            seg_pad_h, seg_pad_w = seg_pad.shape
            h_off = random.randint(0, seg_pad_h - self.crop_size)
            w_off = random.randint(0, seg_pad_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]
            # Generate target maps
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 0
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 0
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        return images, segmentations_half, name
    def __len__(self):
        return len(self.imgs)

class lb_DataGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training

    def __getitem__(self, index):
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # load data
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape[:2]
            pad_h = max(self.crop_size - img_h, 0)
            pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
            pad_w = max(self.crop_size - img_w, 0)
            pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            seg_pad_h, seg_pad_w = seg_pad.shape
            h_off = random.randint(0, seg_pad_h - self.crop_size)
            w_off = random.randint(0, seg_pad_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]
            # Generate target maps
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 0
            seg_half[(seg_half > 4) & (seg_half < 255)] = 1
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 0
            seg_half[(seg_half > 4) & (seg_half < 255)] = 1
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        return images, segmentations_half, name
    def __len__(self):
        return len(self.imgs)

class lub_DataGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training

    def __getitem__(self, index):
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # load data
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape[:2]
            pad_h = max(self.crop_size - img_h, 0)
            pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
            pad_w = max(self.crop_size - img_w, 0)
            pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            seg_pad_h, seg_pad_w = seg_pad.shape
            h_off = random.randint(0, seg_pad_h - self.crop_size)
            w_off = random.randint(0, seg_pad_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]
            # Generate target maps
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        return images, segmentations_half, name
    def __len__(self):
        return len(self.imgs)

class TestGenerator(data.Dataset):

    def __init__(self, root, list_path, crop_size):

        imgs, segs = make_dataset(root, list_path)
        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size

    def __getitem__(self, index):
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)
        ori_size = img.shape

        h, w = seg.shape
        length = max(w, h)
        ratio = self.crop_size / length
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32) - mean
        img = img.transpose((2, 0, 1))
        seg_half = seg.copy()
        seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
        seg_half[(seg_half > 4) & (seg_half < 255)] = 2

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()

        return images, segmentations, segmentations_half, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)


class ReportGenerator(data.Dataset):

    def __init__(self, root, list_path, crop_size):

        imgs, segs = make_dataset(root, list_path)
        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size

    def __getitem__(self, index):
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)
        ori_size = img.shape

        h, w = seg.shape
        length = max(w, h)
        ratio = self.crop_size / length
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32) - mean
        img = img.transpose((2, 0, 1))
        seg_half = seg.copy()
        seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
        seg_half[(seg_half > 4) & (seg_half < 255)] = 2
        seg_full = seg.copy()
        seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        return images, segmentations, segmentations_half, segmentations_full, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dl = DataGenerator('/media/jzzz/Data/Dataset/PascalPersonPart/', './pascal/train_id.txt',
                       crop_size=512,  training=True)

    item = iter(dl)
    for i in range(len(dl)):
        imgs, segs, segs_half, segs_full, idx = next(item)
        pass
