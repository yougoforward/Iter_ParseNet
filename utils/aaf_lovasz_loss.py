from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
import utils.aaf.losses as lossx

class ABRLovaszLoss_backbone_aspp(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszLoss_backbone_aspp, self).__init__()
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
        loss = self.criterion(pred, targets[0])
        # pred = F.softmax(input=pred, dim=1)
        # loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # # half body
        # pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        # pred_hb = F.softmax(input=pred_hb, dim=1)
        # loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
        #                               only_present=self.only_present)
        # # full body
        # pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # pred_fb = F.softmax(input=pred_fb, dim=1)
        # loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
        #                               only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_dsn

class ABRLovaszLoss_backbone(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszLoss_backbone, self).__init__()
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
        # # half body
        # pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        # pred_hb = F.softmax(input=pred_hb, dim=1)
        # loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
        #                               only_present=self.only_present)
        # # full body
        # pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # pred_fb = F.softmax(input=pred_fb, dim=1)
        # loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
        #                               only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_dsn

class ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
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

class ABRLovaszLoss_List(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(ABRLovaszLoss_List, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        loss=[]
        for i in range(len(preds[0])):
            pred = F.interpolate(input=preds[0][i], size=(h, w), mode='bilinear', align_corners=True)
            pred = F.softmax(input=pred, dim=1)
            loss.append(lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present))
        loss = sum(loss)

        # half body
        loss_hb = []
        for i in range(len(preds[1])):
            pred_hb = F.interpolate(input=preds[1][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_hb = F.softmax(input=pred_hb, dim=1)
            loss_hb.append(lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present))
        loss_hb = sum(loss_hb)

        # full body
        loss_fb = []
        for i in range(len(preds[2])):
            pred_fb = F.interpolate(input=preds[2][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_fb = F.softmax(input=pred_fb, dim=1)
            loss_fb.append(lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present))
        loss_fb = sum(loss_fb)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class ABRLovaszLoss_List_final(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(ABRLovaszLoss_List_final, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        final_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # seg loss
        loss=[]
        for i in range(len(preds[3])):
            pred = F.interpolate(input=preds[3][i], size=(h, w), mode='bilinear', align_corners=True)
            pred = F.softmax(input=pred, dim=1)
            loss.append(lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present))
        loss = sum(loss)

        # half body
        loss_hb = []
        for i in range(len(preds[1])):
            pred_hb = F.interpolate(input=preds[1][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_hb = F.softmax(input=pred_hb, dim=1)
            loss_hb.append(lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present))
        loss_hb = sum(loss_hb)

        # full body
        loss_fb = []
        for i in range(len(preds[2])):
            pred_fb = F.interpolate(input=preds[2][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_fb = F.softmax(input=pred_fb, dim=1)
            loss_fb.append(lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present))
        loss_fb = sum(loss_fb)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return final_loss + 0.4*loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class gnn_ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(gnn_ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
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

        # seg loss
        pred = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = loss + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

class AAF_Loss(nn.Module):
    """
    Loss function for multiple outputs
    """

    def __init__(self, ignore_index=255,  only_present=True, num_classes=7):
        super(AAF_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
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
        dec = targets[-1]
        aaf_loss = torch.mean(eloss_1) * self.kld_lambda_1*dec
        aaf_loss += torch.mean(eloss_2) * self.kld_lambda_1*dec
        aaf_loss += torch.mean(eloss_3) * self.kld_lambda_1*dec
        aaf_loss += torch.mean(neloss_1) * self.kld_lambda_2*dec
        aaf_loss += torch.mean(neloss_2) * self.kld_lambda_2*dec
        aaf_loss += torch.mean(neloss_3) * self.kld_lambda_2*dec

        return aaf_loss


class abr_gnn_ABRLovaszLoss2(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True, cls_p=7, cls_h=3, cls_f=2):
        super(abr_gnn_ABRLovaszLoss2, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()
        self.aaf_loss = AAF_Loss(num_classes=cls_p)

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
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + 0.1 * att_bceloss+ self.aaf_loss(preds, targets)
        # return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn

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
