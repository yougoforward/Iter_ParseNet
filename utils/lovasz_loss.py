from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
import utils.aaf.losses as lossx

class CIHPLoss(nn.Module):
    """Lovasz loss for CIHP dataset"""

    def __init__(self, ignore_index=None, only_present=True):
        super(CIHPLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # for train & validation set
        self.weight = torch.FloatTensor([0.81887743, 1.03251272, 0.92667128, 1.19189328, 1.22581131,
                                         0.89328150, 0.96830741, 0.90521306, 1.13456073, 0.93323219,
                                         0.98543486, 1.10501918, 1.06807370, 0.92914351, 0.98171187,
                                         0.97782031, 1.06480612, 1.06545417, 1.09911302, 1.09818779])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.4 * (loss_hb + loss_fb) + 0.4 * loss_dsn


class ReportLovaszLoss(nn.Module):
    """Lovasz loss for Gama process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ReportLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # for train & validation set
        self.weight = torch.FloatTensor([0.79791215, 0.98794882, 0.90357745, 1.08424133, 1.16296374,
                                         0.85149575, 0.99265935, 0.88371943, 1.09656176, 0.89196084,
                                         1.05088595, 1.11189728, 1.06991436, 0.89646903, 0.92704287,
                                         0.92220840, 1.02197269, 1.02266232, 1.03779109, 1.03816560])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.4 * (loss_hb + loss_fb) + 0.4 * loss_dsn


class FinetuneLovaszLoss(nn.Module):
    """Lovasz loss for Gama process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(FinetuneLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # for train & validation set
        self.weight = torch.FloatTensor([0.79791215, 0.98794882, 0.90357745, 1.08424133, 1.16296374,
                                         0.85149575, 0.99265935, 0.88371943, 1.09656176, 0.89196084,
                                         1.05088595, 1.11189728, 1.06991436, 0.89646903, 0.92704287,
                                         0.92220840, 1.02197269, 1.02266232, 1.03779109, 1.03816560])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.4 * (loss_hb + loss_fb)


class TrainValLovaszLoss(nn.Module):
    """Lovasz loss for Gama process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(TrainValLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # for train & validation set
        self.weight = torch.FloatTensor([0.79791215, 0.98794882, 0.90357745, 1.08424133, 1.16296374,
                                         0.85149575, 0.99265935, 0.88371943, 1.09656176, 0.89196084,
                                         1.05088595, 1.11189728, 1.06991436, 0.89646903, 0.92704287,
                                         0.92220840, 1.02197269, 1.02266232, 1.03779109, 1.03816560])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.5 * (loss_hb + loss_fb) + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.2 * loss_dsn


# class GateLovaszLoss(nn.Module):
#     """Lovasz loss for Gating process"""
#
#     def __init__(self, ignore_index=None, only_present=True, weight_rate=1.5, target_rate=0.7):
#         super(GateLovaszLoss, self).__init__()
#         self.weight_rate = weight_rate
#         self.target_rate = target_rate
#         self.ignore_index = ignore_index
#         self.present = only_present
#         self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
#                                          0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
#                                          1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
#                                          0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
#         self.criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
#
#     def forward(self, preds, targets):
#         h, w = targets[0].size(1), targets[0].size(2)
#         # seg loss
#         pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
#         pred = F.softmax(input=pred, dim=1)
#         loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)
#
#         pred_g = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
#         pred_g = F.softmax(input=pred_g, dim=1)
#         loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
#
#         pred_a = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
#         pred_a = F.softmax(input=pred_a, dim=1)
#         loss_a = lovasz_softmax_flat(*flatten_probas(pred_a, targets[0], self.ignore_index), only_present=self.present)
#         # hb loss
#         pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
#         pred_hb = F.softmax(input=pred_hb, dim=1)
#         loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
#                                       only_present=self.present)
#         pred_hb_g = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
#         pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
#         loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
#                                         only_present=self.present)
#         pred_hb_b = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
#         pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
#         loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
#                                         only_present=self.present)
#         pred_hb_a = F.interpolate(input=preds[8], size=(h, w), mode='bilinear', align_corners=True)
#         pred_hb_a = F.softmax(input=pred_hb_a, dim=1)
#         loss_hb_a = lovasz_softmax_flat(*flatten_probas(pred_hb_a, targets[1], self.ignore_index),
#                                         only_present=self.present)
#         # fb loss
#         pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
#         pred_fb = F.softmax(input=pred_fb, dim=1)
#         loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
#                                       only_present=self.present)
#         pred_fb_b = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
#         pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
#         loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
#                                         only_present=self.present)
#         pred_fb_a = F.interpolate(input=preds[9], size=(h, w), mode='bilinear', align_corners=True)
#         pred_fb_a = F.softmax(input=pred_fb_a, dim=1)
#         loss_fb_a = lovasz_softmax_flat(*flatten_probas(pred_fb_a, targets[2], self.ignore_index),
#                                         only_present=self.present)
#         # target rate loss
#         acts = 0
#         activation_rates = preds[-2]
#         for act in activation_rates:
#             acts += torch.pow(self.target_rate - torch.mean(act), 2)
#         acts = torch.mean(acts / len(activation_rates))
#         act_loss = self.weight_rate * acts
#
#         # dsn loss
#         pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
#         loss_dsn = self.criterion(pred_dsn, targets[0])
#         return loss + 0.2 * (loss_hb_g +loss_hb_b+ loss_hb_a + loss_fb_b +loss_fb_a + loss_g+loss_a) + 0.4 * (loss_hb + loss_fb) \
#                + 0.4 * loss_dsn + act_loss

class GateLovaszLoss(nn.Module):
    """Lovasz loss for Gating process"""

    def __init__(self, ignore_index=None, only_present=True, weight_rate=1.5, target_rate=0.7):
        super(GateLovaszLoss, self).__init__()
        self.weight_rate = weight_rate
        self.target_rate = target_rate
        self.ignore_index = ignore_index
        self.present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # target rate loss
        acts = 0
        activation_rates = preds[7]
        for act in activation_rates:
            acts += torch.pow(self.target_rate - torch.mean(act), 2)
        acts = torch.mean(acts / len(activation_rates))
        act_loss = self.weight_rate * acts
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.4 * (loss_hb + loss_fb) \
               + 0.4 * loss_dsn + act_loss

class GamaLovaszLoss(nn.Module):
    """Lovasz loss for Gama process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(GamaLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # for hard example
        self.weight = torch.FloatTensor([0.80595050, 0.99849034, 0.90806005, 1.10437204, 1.17122718,
                                         0.85936701, 1.00941860, 0.89053160, 1.11447017, 0.91007724,
                                         1.03081441, 1.07539078, 1.05182196, 0.90014018, 0.93379550,
                                         0.92896637, 1.04256818, 1.04257404, 1.05865103, 1.05950533])
        # self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
        #                                  0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
        #                                  1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
        #                                  0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)

        pred_g = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_g = F.softmax(input=pred_g, dim=1)
        loss_g = lovasz_softmax_flat(*flatten_probas(pred_g, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)

        pred_hb_g = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_g = F.softmax(input=pred_hb_g, dim=1)
        loss_hb_g = lovasz_softmax_flat(*flatten_probas(pred_hb_g, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.2 * (loss_hb_b + loss_fb_b + loss_hb_g + loss_g) + 0.4 * (loss_hb + loss_fb) + 0.4 * loss_dsn


class BetaLovaszLoss(nn.Module):
    """Lovasz loss for Beta process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(BetaLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.present = only_present
        # self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
        #                                  0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
        #                                  1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
        #                                  0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.weight = torch.FloatTensor([0.89680465, 1.14352656, 1.20982646, 0.99269248,
                                         1.17911144, 1.00641032, 1.47017195, 1.16447113])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.present)
        # hb loss
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.present)
        pred_hb_b = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb_b = F.softmax(input=pred_hb_b, dim=1)
        loss_hb_b = lovasz_softmax_flat(*flatten_probas(pred_hb_b, targets[1], self.ignore_index),
                                        only_present=self.present)
        # fb loss
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.present)
        pred_fb_b = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb_b = F.softmax(input=pred_fb_b, dim=1)
        loss_fb_b = lovasz_softmax_flat(*flatten_probas(pred_fb_b, targets[2], self.ignore_index),
                                        only_present=self.present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * (loss_hb + loss_fb) + 0.2 * (loss_hb_b + loss_fb_b) + 0.4 * loss_dsn


class BiRNN_tree_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(BiRNN_tree_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss2 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb2 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb2 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # part_node
        #onehot targets part

        p_cls = len(preds[6])+1
        y = torch.eye(p_cls).cuda()
        target_p_node = targets[0]
        target_p_node[target_p_node==255]=0
        targets_p_node_list = list(torch.split(y[target_p_node], 1, dim=-1))
        for i in range(0,len(targets_p_node_list)):
            targets_p_node_list[i] = targets_p_node_list[i].squeeze(-1)
            targets_p_node_list[i][targets[0]==255]=255
        #remove background
        targets_p_node_list_nob = [targets_p_node_list[i] for i in range(1, p_cls)]
        pred_p_node_list = [F.interpolate(input=preds[6][i], size=(h, w), mode='bilinear', align_corners=True) for i in range(0, len(preds[6]))]
        pred_p_node_list = [F.softmax(input=pred_p_node_list[i], dim=1) for i in range(0, len(preds[6]))]
        p_node_loss_list = [lovasz_softmax_flat(*flatten_probas(pred_p_node_list[i], targets_p_node_list_nob[i], self.ignore_index),
                                       only_present=self.only_present) for i in range(0, len(preds[6]))]

        p_node_loss = p_node_loss_list[0]
        for i in range(1, len(p_node_loss_list)):
            p_node_loss += p_node_loss_list[i]

        # half node
        # onehot targets half
        # bs, h, w= targets[1].size()
        h_cls = len(preds[7])+1
        y2 = torch.eye(h_cls).cuda()
        target_h_node=targets[1]
        target_h_node[target_h_node == 255] = 0
        targets_h_node_list = list(torch.split(y2[target_h_node.long()], 1, dim=-1))

        # y_onehot = torch.FloatTensor(bs, h, w, h_cls).cuda()
        # y_onehot.zero_()
        # y_onehot = y_onehot.scatter_(3, target_h_node.view(bs, h, w, 1).long(), 1)
        # targets_h_node_list = list(torch.split(y_onehot, 1, dim=-1))

        for i in range(0, len(targets_h_node_list)):
            targets_h_node_list[i] = targets_h_node_list[i].squeeze(-1)
            targets_h_node_list[i][targets[1] == 255] = 255
        # remove background
        targets_h_node_list_nob = [targets_h_node_list[i] for i in range(1, h_cls)]

        pred_h_node_list = [F.interpolate(input=preds[7][i], size=(h, w), mode='bilinear', align_corners=True) for i in
                            range(0, len(preds[7]))]
        pred_h_node_list = [F.softmax(input=pred_h_node_list[i], dim=1) for i in range(0, len(preds[7]))]
        h_node_loss_list = [lovasz_softmax_flat(*flatten_probas(pred_h_node_list[i], targets_h_node_list_nob[i], self.ignore_index),
                                                only_present=self.only_present) for i in range(0, len(preds[7]))]
        h_node_loss=h_node_loss_list[0]
        for i in range(1, len(h_node_loss_list)):
            h_node_loss+=h_node_loss_list[i]

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb +0.4*loss2 + 0.2 * loss_hb2 + 0.2 * loss_fb2 + 0.4*p_node_loss + 0.4*h_node_loss + 0.4 * loss_dsn

class treeRNN_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(treeRNN_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # part_node
        #onehot targets part

        p_cls = len(preds[-3])+1
        y = torch.eye(p_cls).cuda()
        target_p_node = targets[0]
        target_p_node[target_p_node==255]=0
        targets_p_node_list = list(torch.split(y[target_p_node], 1, dim=-1))
        for i in range(0,len(targets_p_node_list)):
            targets_p_node_list[i] = targets_p_node_list[i].squeeze(-1)
            targets_p_node_list[i][targets[0]==255]=255
        #remove background
        targets_p_node_list_nob = [targets_p_node_list[i] for i in range(1, p_cls)]
        pred_p_node_list = [F.interpolate(input=preds[-3][i], size=(h, w), mode='bilinear', align_corners=True) for i in range(0, len(preds[-3]))]
        pred_p_node_list = [F.softmax(input=pred_p_node_list[i], dim=1) for i in range(0, len(preds[-3]))]
        p_node_loss_list = [lovasz_softmax_flat(*flatten_probas(pred_p_node_list[i], targets_p_node_list_nob[i], self.ignore_index),
                                       only_present=self.only_present) for i in range(0, len(preds[-3]))]

        p_node_loss = p_node_loss_list[0]
        for i in range(1, len(p_node_loss_list)):
            p_node_loss += p_node_loss_list[i]

        # half node
        # onehot targets half
        # bs, h, w= targets[1].size()
        h_cls = len(preds[-2])+1
        y2 = torch.eye(h_cls).cuda()
        target_h_node=targets[1]
        target_h_node[target_h_node == 255] = 0
        targets_h_node_list = list(torch.split(y2[target_h_node.long()], 1, dim=-1))

        # y_onehot = torch.FloatTensor(bs, h, w, h_cls).cuda()
        # y_onehot.zero_()
        # y_onehot = y_onehot.scatter_(3, target_h_node.view(bs, h, w, 1).long(), 1)
        # targets_h_node_list = list(torch.split(y_onehot, 1, dim=-1))

        for i in range(0, len(targets_h_node_list)):
            targets_h_node_list[i] = targets_h_node_list[i].squeeze(-1)
            targets_h_node_list[i][targets[1] == 255] = 255
        # remove background
        targets_h_node_list_nob = [targets_h_node_list[i] for i in range(1, h_cls)]

        pred_h_node_list = [F.interpolate(input=preds[-2][i], size=(h, w), mode='bilinear', align_corners=True) for i in
                            range(0, len(preds[-2]))]
        pred_h_node_list = [F.softmax(input=pred_h_node_list[i], dim=1) for i in range(0, len(preds[-2]))]
        h_node_loss_list = [lovasz_softmax_flat(*flatten_probas(pred_h_node_list[i], targets_h_node_list_nob[i], self.ignore_index),
                                                only_present=self.only_present) for i in range(0, len(preds[-2]))]
        h_node_loss=h_node_loss_list[0]
        for i in range(1, len(h_node_loss_list)):
            h_node_loss+=h_node_loss_list[i]

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4*p_node_loss + 0.4*h_node_loss + 0.4 * loss_dsn


class BiRNN_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(BiRNN_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss2 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb2 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb2 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb +0.4*loss + 0.2 * loss_hb + 0.2 * loss_fb + 0.4 * loss_dsn

class iter_birnn_tree_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(iter_birnn_tree_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        bu_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        bu_loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        bu_loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        td_loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        td_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)


        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.4*(bu_loss+bu_loss_hb+bu_loss_fb+td_loss_hb+td_loss)

class gnn_final_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(gnn_final_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss_2 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb_2 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb_2 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss_3 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb_3 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[8], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb_3 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb +0.2*(loss_2+loss_hb_2+loss_fb_2+loss_3+loss_hb_3+loss_fb_3)+0.4 * loss_dsn

class att_gnn_iter_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(att_gnn_iter_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss2 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb2 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb2 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss3 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                    only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb3 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                       only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[8], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb3 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                       only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.cls_p)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.cls_p):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            one_hot_pb_list[i][targets[0] == 255] = 255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            one_hot_hb_list[i][targets[1] == 255] = 255

        # labels_f = targets[2]
        # one_label_f = labels_f.clone().long()
        # one_label_f[one_label_f == 255] = 0
        # one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        # one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))

        # p node decomp att list supervision
        p_node_loss_all = []
        for i in range(self.cls_p-1):
            p_node_preds = preds[9][i]
            p_node_preds = F.interpolate(input=p_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            p_node_loss_all.append(lovasz_softmax(p_node_preds, one_hot_pb_list[i+1], classes= [1], per_image=False, ignore=self.ignore_index))
        p_node_loss = sum(p_node_loss_all)

        # h node decomp att supervision
        h_node_loss_all = []
        for i in range(self.cls_h-1):
            h_node_preds = preds[10][i]
            h_node_preds = F.interpolate(input=h_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            h_node_loss_all.append(lovasz_softmax(h_node_preds, one_hot_hb_list[i+1], classes= [1], per_image=False, ignore=self.ignore_index))
        h_node_loss = sum(h_node_loss_all)

        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.2*(loss2+loss_hb2+loss_fb2) +0.2*(loss3+loss_hb3+loss_fb3) + 0.2*(p_node_loss+h_node_loss) + 0.2 * loss_dsn

class abr_gnn_ABRLovaszLoss3(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss3, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb + lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn


class abr_gnn_ABRLovaszLoss2(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss2, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb + lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            # one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0, self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            # one_hot_fb_list[i][targets[2]==255]=255
        # #
        ignore = (targets[0]!=255).float().unsqueeze(1)
        #
        att_onehot = torch.stack(one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        att_bceloss = torch.mean(self.bceloss(F.interpolate(preds[6], size=(h, w), mode='bilinear', align_corners=True), att_onehot)*ignore)

        # node_onehot = torch.stack(one_hot_fb_list+one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        # node_bceloss = torch.mean(self.bceloss(F.interpolate(self.sigmoid(preds[6]), size=(h, w), mode='bilinear', align_corners=True), node_onehot)*ignore)
        # att_bceloss = att_bceloss+ node_bceloss

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1 * att_bceloss
        # return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn
class abr_gnn_ABRLovaszLoss5(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss5, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb + lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb + lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[8], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            # one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0, self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            # one_hot_fb_list[i][targets[2]==255]=255
        # #
        ignore = (targets[0]!=255).float().unsqueeze(1)
        #
        att_onehot = torch.stack(one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        att_bceloss = torch.mean(self.bceloss(F.interpolate(preds[-2], size=(h, w), mode='bilinear', align_corners=True), att_onehot)*ignore)

        # node_onehot = torch.stack(one_hot_fb_list+one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        # node_bceloss = torch.mean(self.bceloss(F.interpolate(self.sigmoid(preds[6]), size=(h, w), mode='bilinear', align_corners=True), node_onehot)*ignore)
        # att_bceloss = att_bceloss+ node_bceloss

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1 * att_bceloss
        # return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class abr_gnn_ABRLovaszLoss4(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss4, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes).float()
        # one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        # for i in range(0, self.num_classes):
        #     one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
        #     # one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h).float()
        # one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        # for i in range(0, self.cls_h):
        #     one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
        #     # one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f).float()
        # one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        # for i in range(0, self.cls_f):
        #     one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
        #     # one_hot_fb_list[i][targets[2]==255]=255
        ignore = (targets[0]!=255).float().unsqueeze(1)
        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        loss = loss + torch.mean(self.bceloss(pred, one_hot_lab_p.permute(0,3,1,2))*ignore)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        loss_hb = loss_hb + torch.mean(self.bceloss(pred_hb, one_hot_lab_h.permute(0,3,1,2))*ignore)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + torch.mean(self.bceloss(pred_fb, one_hot_lab_f.permute(0,3,1,2))*ignore)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class abr_gnn_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index),
                                          only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb + lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                                only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb + lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                                only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            # one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0, self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            # one_hot_fb_list[i][targets[2]==255]=255
        #
        ignore = (targets[0]!=255).float().unsqueeze(1)
        #
        att_onehot = torch.stack(one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        att_bceloss = torch.mean(self.bceloss(F.interpolate(preds[7], size=(h, w), mode='bilinear', align_corners=True), att_onehot)*ignore)

        node_onehot = torch.stack(one_hot_fb_list+one_hot_hb_list[1:]+one_hot_pb_list[1:], dim=1).float()
        node_bceloss = torch.mean(self.bceloss(F.interpolate(self.sigmoid(preds[6]), size=(h, w), mode='bilinear', align_corners=True), node_onehot)*ignore)
        att_bceloss = att_bceloss+ node_bceloss

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1 * att_bceloss

        # return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class gnn_iter_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(gnn_iter_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss+lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = loss_hb+lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = loss_fb+lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn
class hr_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(hr_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        return loss + 0.4 * loss_hb + 0.4 * loss_fb

class ppss_AAF_Loss(nn.Module):
    """
    Loss function for multiple outputs
    """

    def __init__(self, ignore_index=255,  only_present=True, num_classes=7):
        super(ppss_AAF_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.89680465, 1.14352656, 1.20982646, 0.99269248,
                                         1.17911144, 1.00641032, 1.47017195, 1.16447113])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

        self.num_classes=num_classes
        self.kld_margin=3.0
        self.kld_lambda_1=1.0
        self.kld_lambda_2=1.0
        self.dec = 1e-3
        # self.dec = 1e-2
        self.softmax = nn.Softmax(dim=1)
        self.w_edge = nn.Parameter(torch.zeros(1,1,1,self.num_classes,1,3))
        self.w_edge_softmax = nn.Softmax(dim=-1)
        self.w_not_edge = nn.Parameter(torch.zeros(1, 1, 1, self.num_classes, 1, 3))
        self.w_not_edge_softmax = nn.Softmax(dim=-1)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])



        #aaf loss
        labels = targets[0]
        one_label=labels.clone()
        one_label[one_label==255]=0
        one_hot_lab=F.one_hot(one_label, num_classes=self.num_classes)

        targets_p_node_list = list(torch.split(one_hot_lab,1, dim=3))
        for i in range(self.num_classes):
            targets_p_node_list[i] = targets_p_node_list[i].squeeze(-1)
            targets_p_node_list[i][labels==255]=255
        one_hot_lab = torch.stack(targets_p_node_list, dim=-1)

        prob = pred
        w_edge = self.w_edge_softmax(self.w_edge)
        w_not_edge = self.w_not_edge_softmax(self.w_not_edge)

        # w_edge_shape=list(w_edge.shape)
        # Apply AAF on 3x3 patch.
        eloss_1, neloss_1 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         1,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 0],
                                                         w_not_edge[..., 0])
        # Apply AAF on 5x5 patch.
        eloss_2, neloss_2 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         2,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 1],
                                                         w_not_edge[..., 1])
        # Apply AAF on 7x7 patch.
        eloss_3, neloss_3 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         3,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 2],
                                                         w_not_edge[..., 2])
        dec = self.dec
        aaf_loss = torch.mean(eloss_1) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_2) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_3) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(neloss_1) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_2) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_3) * self.kld_lambda_2 * dec

        # return torch.stack([loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn, aaf_loss], dim=0)
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + aaf_loss

class PPSS_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(PPSS_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.89680465, 1.14352656, 1.20982646, 0.99269248,
                                         1.17911144, 1.00641032, 1.47017195, 1.16447113])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class dp_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(dp_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss+ 0.4*lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)

        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class ABRLovaszLoss2(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszLoss2, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss_ce = self.criterion(pred, targets[0])

        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss_hb_ce = self.criterion2(pred_hb, targets[1].long())

        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss_fb_ce = self.criterion2(pred_fb, targets[2].long())

        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return  loss_ce+0.4*loss_hb_ce+0.4*loss_fb_ce+loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class du_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(du_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.l2loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.shuffle = torch.nn.PixelShuffle(8)
        self.r =8
    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        b,c,ph,pw=preds[0].size()
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        # pred = preds[0]
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        W = preds[3]
        P = preds[4]

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=c).permute(0,3,1,2).float() # n, c, h,w
        # one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        # for i in range(0, self.num_classes):
        #     one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0] == 255] = 255

        node_weight = (targets[0]!=255).float().unsqueeze(1)
        ex_one_hot_lab_p = F.interpolate(input=one_hot_lab_p, size=(ph, pw), mode='bilinear', align_corners=True)
        # ex_one_hot_lab_p = one_hot_lab_p
        # inverse shuffle

        out_channel = c * (self.r ** 2)
        out_h = ph // self.r
        out_w = pw // self.r
        fm_view = ex_one_hot_lab_p.contiguous().view(b, c, out_h, self.r, out_w, self.r)
        fm_prime = fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).permute(0,2,3,1)
        es_targets = torch.matmul(torch.matmul(fm_prime, P), W)
        es_targets = self.shuffle(es_targets.permute(0,3,1,2))
        es_targets = F.interpolate(input=es_targets, size=(h, w), mode='bilinear', align_corners=True)
        reg_loss = self.l2loss(es_targets*node_weight,one_hot_lab_p*node_weight)

        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + reg_loss

class nodesoftmax_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(nodesoftmax_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = 7
        self.cls_h = 3
        self.cls_f = 2

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0,self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0,self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0,self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            one_hot_fb_list[i][targets[2]==255]=255


        # pnode list supervision
        p_node_loss_all = []
        plist_len = len(preds[3])
        for i in range(plist_len):
            p_node_preds = preds[3][i]
            p_node_preds = F.interpolate(input=p_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            p_node_preds = F.softmax(input=p_node_preds, dim=1)
            p_node_loss = lovasz_softmax_flat(*flatten_probas(p_node_preds, targets[0], self.ignore_index),
                                       only_present=self.only_present)

            p_node_loss_all.append(p_node_loss)
        p_node_loss = p_node_loss_all[0]
        for i in range(1, len(p_node_loss_all)):
            p_node_loss += p_node_loss_all[i]

        # h node supervision
        h_node_loss_all = []
        hlist_len = len(preds[4])
        for i in range(hlist_len):
            h_node_preds = preds[4][i]
            h_node_preds = F.interpolate(input=h_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            h_node_preds = F.softmax(input=h_node_preds, dim=1)
            h_node_loss = lovasz_softmax_flat(*flatten_probas(h_node_preds, targets[1], self.ignore_index),
                                              only_present=self.only_present)
            h_node_loss_all.append(h_node_loss)
        h_node_loss = h_node_loss_all[0]
        for i in range(1, len(h_node_loss_all)):
            h_node_loss += h_node_loss_all[i]


        # f node supervision
        f_node_loss_all = []
        flist_len = len(preds[5])
        for i in range(flist_len):
            f_node_preds = preds[5][i]
            f_node_preds = F.interpolate(input=f_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            f_node_preds = F.softmax(input=f_node_preds, dim=1)
            f_node_loss = lovasz_softmax_flat(*flatten_probas(f_node_preds, targets[2], self.ignore_index),
                                              only_present=self.only_present)
            f_node_loss_all.append(f_node_loss)
        f_node_loss = f_node_loss_all[0]
        for i in range(1, len(f_node_loss_all)):
            f_node_loss += f_node_loss_all[i]
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1*(p_node_loss + h_node_loss + f_node_loss)

class nodebce_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(nodebce_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = 7
        self.cls_h = 3
        self.cls_f = 2

        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == 255] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0,self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            one_hot_pb_list[i][targets[0]==255]=255

        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == 255] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0,self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            one_hot_hb_list[i][targets[1]==255]=255

        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == 255] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0,self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            one_hot_fb_list[i][targets[2]==255]=255


        # pnode list supervision
        p_node_loss_all = []
        plist_len = len(preds[3])
        for i in range(plist_len):
            p_node_preds = preds[3][i]
            p_node_preds = F.interpolate(input=p_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            p_node_list = list(torch.split(p_node_preds, 1, dim=1))
            # p_node_sigmoid_list = [F.sigmoid(p_node_list[j].squeeze(dim=1)) for j in range(self.num_classes)]
            # p_node_loss_list = [lovasz_softmax(p_node_sigmoid_list[k], one_hot_pb_list[k], classes= [1], per_image=False, ignore=self.ignore_index) for k in range(self.num_classes)]

            p_node_loss_list = [lovasz_hinge(p_node_list[k].squeeze(dim=1), one_hot_pb_list[k], per_image=False, ignore=self.ignore_index) for k in range(self.num_classes)]
            p_node_loss = p_node_loss_list[0]
            for i in range(1, len(p_node_loss_list)):
                p_node_loss += p_node_loss_list[i]
            p_node_loss_all.append(p_node_loss)
        p_node_loss = p_node_loss_all[0]
        for i in range(1, len(p_node_loss_all)):
            p_node_loss += p_node_loss_all[i]

        # h node supervision
        h_node_loss_all = []
        hlist_len = len(preds[4])
        for i in range(hlist_len):
            h_node_preds = preds[4][i]
            h_node_preds = F.interpolate(input=h_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            h_node_list = list(torch.split(h_node_preds, 1, dim=1))
            # h_node_sigmoid_list = [F.sigmoid(h_node_list[j].squeeze(dim=1)) for j in range(self.cls_h)]
            # h_node_loss_list = [lovasz_softmax(h_node_sigmoid_list[k], one_hot_hb_list[k], classes= [1], per_image=False, ignore=self.ignore_index) for k in range(self.num_classes)]

            h_node_loss_list = [
                lovasz_hinge(h_node_list[k], one_hot_hb_list[k], per_image=True, ignore=self.ignore_index)
                for k in range(self.cls_h)]
            h_node_loss = h_node_loss_list[0]
            for i in range(1, len(h_node_loss_list)):
                h_node_loss += h_node_loss_list[i]
            h_node_loss_all.append(h_node_loss)
        h_node_loss = h_node_loss_all[0]
        for i in range(1, len(h_node_loss_all)):
            h_node_loss += h_node_loss_all[i]


        # f node supervision
        f_node_loss_all = []
        flist_len = len(preds[5])
        for i in range(flist_len):
            f_node_preds = preds[5][i]
            f_node_preds = F.interpolate(input=f_node_preds, size=(h, w), mode='bilinear', align_corners=True)
            f_node_list = list(torch.split(f_node_preds, 1, dim=1))
            # f_node_sigmoid_list = [F.sigmoid(f_node_list[j].squeeze(dim=1)) for j in range(self.cls_f)]
            # f_node_loss_list = [lovasz_softmax(f_node_sigmoid_list[k], one_hot_fb_list[k], classes=[1], per_image=False, ignore=self.ignore_index) for k in range(self.num_classes)]

            f_node_loss_list = [
                lovasz_hinge(f_node_list[k], one_hot_fb_list[k], per_image=True, ignore=self.ignore_index)
                for k in range(self.cls_f)]
            f_node_loss = f_node_loss_list[0]
            for i in range(1, len(f_node_loss_list)):
                f_node_loss += f_node_loss_list[i]
            f_node_loss_all.append(f_node_loss)
        f_node_loss = f_node_loss_all[0]
        for i in range(1, len(f_node_loss_all)):
            f_node_loss += f_node_loss_all[i]
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1*(p_node_loss/self.num_classes + h_node_loss/self.cls_h + f_node_loss/self.cls_f)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.ep = 1e-10

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        # inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)+self.ep

        loss = inter / union

        ## Return average loss over classes and batch
        return 1-loss.mean()

class HIOU_Loss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(HIOU_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.miouloss = mIoULoss(weight=None, size_average=True, n_classes=1)
        self.upper_parts = [1,2,3,4]
        self.lower_parts = [5,6]
        self.num_classes = 7
        self.cls_h = 3
        self.cls_f = 2

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        ############hierarchical iou loss#####################
        pred_fb_list = list(torch.split(pred_fb, 1, dim=1))
        pred_hb_list = list(torch.split(pred_hb, 1, dim=1))
        pred_pb_list = list(torch.split(pred, 1, dim=1))

        # labels_p = targets[0]
        # one_label_p = labels_p.clone().long()
        # one_label_p[one_label_p == 255] = 0
        # one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        # one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        # for i in range(0, self.num_classes):
        #     one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
        #     one_hot_pb_list[i][targets[0] == 255] = 255
        #
        # labels_h = targets[1]
        # one_label_h = labels_h.clone().long()
        # one_label_h[one_label_h == 255] = 0
        # one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        # one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        # for i in range(0, self.cls_h):
        #     one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
        #     one_hot_hb_list[i][targets[1] == 255] = 255
        #
        # labels_f = targets[2]
        # one_label_f = labels_f.clone().long()
        # one_label_f[one_label_f == 255] = 0
        # one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        # one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        # for i in range(0, self.cls_f):
        #     one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
        #     one_hot_fb_list[i][targets[2] == 255] = 255



        # full boady and half body iou loss
        pred_hb_full = torch.max(torch.cat([pred_hb_list[1], pred_hb_list[2]], dim=1), dim=1, keepdim=True)[0]
        iouloss_fh = self.miouloss(pred_hb_full, pred_fb_list[1])
        # iouloss_fh = lovasz_softmax_flat(*flatten_probas(pred_hb_full, one_hot_fb_list[1], self.ignore_index),
        #                               only_present=self.only_present)
        # iouloss_fh = lovasz_softmax(pred_hb_full, one_hot_fb_list[1], classes=[1], per_image=False,
        #                             ignore=self.ignore_index)

        # half body and parts iou loss
        pred_pb_upper_list = [pred_pb_list[i] for i in self.upper_parts]
        pred_pb_lower_list = [pred_pb_list[i] for i in self.lower_parts]
        pred_pb_upper = torch.max(torch.cat(pred_pb_upper_list, dim=1), dim=1, keepdim=True)[0]
        pred_pb_lower = torch.max(torch.cat(pred_pb_lower_list, dim=1), dim=1, keepdim=True)[0]
        iouloss_hp_upper = self.miouloss(pred_hb_list[1], pred_pb_upper)
        iouloss_hp_lower = self.miouloss(pred_hb_list[2], pred_pb_lower)

        pred_pb = torch.max(torch.cat(pred_pb_list, dim=1), dim=1, keepdim=True)[0]
        iouloss_pb = self.miouloss(pred_pb, pred_fb_list[1])

        # iouloss_hp_upper = lovasz_softmax_flat(*flatten_probas(pred_pb_upper, one_hot_hb_list[1], self.ignore_index),
        #                                  only_present=self.only_present)
        # iouloss_hp_lower = lovasz_softmax_flat(*flatten_probas(pred_pb_lower, one_hot_hb_list[2], self.ignore_index),
        #                                  only_present=self.only_present)
        # iouloss_hp_upper = lovasz_softmax(pred_pb_upper, one_hot_hb_list[1], classes=[1], per_image=False,
        #                             ignore=self.ignore_index)
        # iouloss_hp_lower = lovasz_softmax(pred_pb_lower, one_hot_hb_list[2], classes=[1], per_image=False,
        #                             ignore=self.ignore_index)
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1*(iouloss_fh+iouloss_hp_upper+iouloss_hp_lower+ iouloss_pb)

class AAF_Loss(nn.Module):
    """
    Loss function for multiple outputs
    """

    def __init__(self, ignore_index=255,  only_present=True, num_classes=7):
        super(AAF_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

        self.num_classes=num_classes
        self.kld_margin=3.0
        self.kld_lambda_1=1.0
        self.kld_lambda_2=1.0
        self.dec = 1e-3
        # self.dec = 1e-2
        self.softmax = nn.Softmax(dim=1)
        self.w_edge = nn.Parameter(torch.zeros(1,1,1,self.num_classes,1,3))
        self.w_edge_softmax = nn.Softmax(dim=-1)
        self.w_not_edge = nn.Parameter(torch.zeros(1, 1, 1, self.num_classes, 1, 3))
        self.w_not_edge_softmax = nn.Softmax(dim=-1)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])



        #aaf loss
        labels = targets[0]
        one_label=labels.clone()
        one_label[one_label==255]=0
        one_hot_lab=F.one_hot(one_label, num_classes=self.num_classes)

        targets_p_node_list = list(torch.split(one_hot_lab,1, dim=3))
        for i in range(self.num_classes):
            targets_p_node_list[i] = targets_p_node_list[i].squeeze(-1)
            targets_p_node_list[i][labels==255]=255
        one_hot_lab = torch.stack(targets_p_node_list, dim=-1)

        prob = pred
        w_edge = self.w_edge_softmax(self.w_edge)
        w_not_edge = self.w_not_edge_softmax(self.w_not_edge)

        # w_edge_shape=list(w_edge.shape)
        # Apply AAF on 3x3 patch.
        eloss_1, neloss_1 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         1,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 0],
                                                         w_not_edge[..., 0])
        # Apply AAF on 5x5 patch.
        eloss_2, neloss_2 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         2,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 1],
                                                         w_not_edge[..., 1])
        # Apply AAF on 7x7 patch.
        eloss_3, neloss_3 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         3,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 2],
                                                         w_not_edge[..., 2])
        dec = self.dec
        aaf_loss = torch.mean(eloss_1) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_2) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_3) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(neloss_1) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_2) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_3) * self.kld_lambda_2 * dec

        # return torch.stack([loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn, aaf_loss], dim=0)
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + aaf_loss


class lub_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(lub_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # half body
        pred_hb = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[0], self.ignore_index),
                                      only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        return loss_hb + 0.4 * loss_dsn



class se_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, nclass=20, ignore_index=None, only_present=True):
        super(se_ABRLovaszLoss, self).__init__()
        self.nclass = nclass
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.bceloss = BCELoss(weight=None, size_average=True)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        # se loss
        se_target = self._get_batch_label_vector(targets[0], nclass=self.nclass).type_as(preds[-2])
        loss_se = self.bceloss(torch.sigmoid(preds[-2]), se_target)

        return loss + 0.4 * loss_hb + 0.4*loss_fb + 0.4*loss_dsn + 0.2*loss_se

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect

class iter_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(iter_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss1 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[4], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb1 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[5], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb1 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)

        # # seg loss
        # pred = F.interpolate(input=preds[6], size=(h, w), mode='bilinear', align_corners=True)
        # pred = F.softmax(input=pred, dim=1)
        # loss2 = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[7], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb2 = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[8], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb2 = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # seg loss
        pred = F.interpolate(input=preds[9], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss_final = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss_final + 0.4 * (loss + loss_hb + loss_fb)+ 0.4*(loss1 + loss_hb1 + loss_fb1 + loss_hb2 + loss_fb2) + 0.4*loss_dsn


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz loss for Deep Supervision"""

    def __init__(self, ignore_index=None, only_present=False, per_image=False):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.per_image = per_image
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def forward(self, preds, targets):
        h, w = targets.size(1), targets.size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        if self.per_image:
            loss = mean(lovasz_softmax_flat(*flatten_probas(pre.unsqueeze(0), tar.unsqueeze(0), self.ignore_index),
                                            only_present=self.only_present) for pre, tar in zip(pred, targets))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(pred, targets, self.ignore_index),
                                       only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets)
        return loss + 0.4 * loss_dsn


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat_ori(*flatten_probas_ori(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat_ori(*flatten_probas_ori(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat_ori(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas_ori(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(preds, targets, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      :param preds: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      :param targets: [P] Tensor, ground truth labels (between 0 and C - 1)
      :param only_present: average only on classes present in ground truth
    """
    if preds.numel() == 0:
        # only void pixels, the gradients should be 0
        return preds * 0.

    C = preds.size(1)
    losses = []
    for c in range(C):
        fg = (targets == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - preds[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(preds, targets, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = preds.size()
    preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    targets = targets.view(-1)
    if ignore is None:
        return preds, targets
    valid = (targets != ignore)
    vprobas = preds[valid.nonzero().squeeze()]
    vlabels = targets[valid]
    return vprobas, vlabels

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=True, empty=0):
    """
    nan mean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def isnan(x):
    return x != x
