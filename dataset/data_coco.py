import os
import os.path
import random

import cv2
import numpy as np
import torch.utils.data as data


# ###### Data loading #######
def make_dataset(root, lst):
    # append all index
    fid = open(lst, 'r')
    imgs, segs = [], []
    for line in fid.readlines():
        idx = line.strip().split(' ')
        image_path = os.path.join(root, idx[0])
        seg_path = os.path.join(root, idx[1])
        imgs.append(image_path)
        segs.append(seg_path)
    return imgs, segs


# ###### val resize & crop ######
def scale_crop(img, seg, crop_size):
    oh, ow = seg.shape
    pad_h = max(0, crop_size - oh)
    pad_w = max(0, crop_size - ow)
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        seg_pad = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=255)
    else:
        img_pad, seg_pad = img, seg

    img = np.asarray(img_pad[0: crop_size, 0: crop_size], np.float32)
    seg = np.asarray(seg_pad[0: crop_size, 0: crop_size], np.float32)

    return img, seg


class DatasetGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training

    def __getitem__(self, index):
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg_in = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            if flip == -1:
                seg = seg_in[:, ::-1]
            else:
                seg = seg_in
            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape
            pad_h = max(self.crop_size - img_h, 0)
            pad_w = max(self.crop_size - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            img_h, img_w = seg_pad.shape
            h_off = random.randint(0, img_h - self.crop_size)
            w_off = random.randint(0, img_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            img = img.transpose((2, 0, 1))

        else:
            h, w = seg_in.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg_in, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))

        images = img.copy()
        segmentations = seg.copy()

        return images, segmentations, name

    def __len__(self):
        return len(self.imgs)
