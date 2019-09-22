import torch
import torch.nn as nn
from torch.nn import functional as F

torch_ver = torch.__version__[:3]
# assert torch_ver == '0.4'

import utils.aaf.losses as lossx

class ABRCELoss(nn.Module):
    """CE loss for Alpha process"""

    def __init__(self, ignore_index=255):
        super(ABRCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.weight_hb = torch.FloatTensor([0.99783757, 1.0101299, 1.1039613])
        self.weight_fb = torch.FloatTensor([0.99783757, 1.00252392])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)
        self.criterion_hb = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight_hb)
        self.criterion_fb = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight_fb)

    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(pred, target[0])
        # half body loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss_hb = self.criterion_hb(pred_hb, target[1])
        # full body loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss_fb = self.criterion_fb(pred_fb, target[2])
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, target[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn


class SegmentationMultiLoss(nn.Module):
    """
    Loss function for multiple outputs
    """

    def __init__(self, ignore_index=255):
        super(SegmentationMultiLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(pred, target)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, target)
        return loss + 0.4 * loss_dsn


