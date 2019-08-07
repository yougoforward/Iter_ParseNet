import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from . import convolutional_rnn
from modules.convlstm import plus_tree_ConvLSTMCell , max_tree_ConvLSTMCell, tree_ConvLSTMCell

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        _, _, th, tw = xl.size()
        xt_fea = self.conv1(xt)
        xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        x_seg = self.conv4(x_fea)

        xt_seg = F.interpolate(x_seg, size=(h, w), mode='bilinear', align_corners=True)

        return x_seg, xt_fea, xt_seg


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

        self.conv2 =nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)

        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        xfuse = self.conv1(xfuse)
        output = self.conv2(xfuse)
        return output, xfuse

class GFF(nn.Module):
    def __init__(self, indim1, indim2, ncls, reduction=16):
        super(GFF, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(indim1, indim1//reduction, kernel_size=1, padding=0, stride=1, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(indim1//reduction, ncls, kernel_size=1, padding=0, stride=1, bias=True),
                                   nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(indim2, indim2 // reduction, kernel_size=1, padding=0, stride=1,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(indim2 // reduction, ncls, kernel_size=1, padding=0, stride=1,
                                             bias=True),
                                   nn.Sigmoid())

        self.conv3 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(indim1+indim2, (indim1+indim2) // reduction, kernel_size=1, padding=0, stride=1,
                                             bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d((indim1+indim2) // reduction, ncls, kernel_size=1, padding=0, stride=1, bias=True),
                                   nn.Sigmoid())

    def forward(self, x1, x2, seg1, seg2):

        x1_se = self.conv1(x1)
        x2_se = self.conv2(x2)
        x3_se = self.conv3(torch.cat([x1,x2], dim=1))
        seg = seg1 + x3_se*(seg1*x1_se) + (1-x3_se)*(1-x1_se)*(seg2*x2_se)
        return seg

class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))
        self.conv2 = nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_fb * skip
        xfuse = self.conv1(xfuse)
        output = self.conv2(xfuse)
        return output, xfuse


class Final_DecoderModule(nn.Module):

    def __init__(self, cls_p=7, cls_h=3, cls_f=2):
        super(Final_DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(118, 128, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(128), nn.ReLU(inplace=False),
                                   nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(128), nn.ReLU(inplace=False))

        self.conv3 = nn.Conv2d(10*cls_p, cls_p, kernel_size=1, padding=0, dilation=1, bias=True, groups=cls_p)
        self.conv4 = nn.Conv2d(10*cls_h, cls_h, kernel_size=1, padding=0, dilation=1, bias=True, groups=cls_h)
        self.conv5 = nn.Conv2d(10*cls_f, cls_f, kernel_size=1, padding=0, dilation=1, bias=True, groups=cls_f)
        self.conv6 = nn.Conv2d(10*cls_h, cls_h, kernel_size=1, padding=0, dilation=1, bias=True, groups=cls_h)
        self.conv7 = nn.Conv2d(10*cls_p, cls_p, kernel_size=1, padding=0, dilation=1, bias=True, groups=cls_p)

        self.conv_final = nn.Conv2d(128, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, bu_part_fea, bu_half_fea, bu_full_fea, td_half_fea, td_part_fea, xl):
        _, _, th, tw = xl.size()

        xp = F.interpolate(td_part_fea, size=(th, tw), mode='bilinear', align_corners=True)
        # xh = F.interpolate(xh, size=(th, tw), mode='bilinear', align_corners=True)
        # xf = F.interpolate(xf, size=(th, tw), mode='bilinear', align_corners=True)

        xl = self.conv1(xl)

        xpl = torch.cat([xp, xl], dim=1)
        xpl = self.conv2(xpl)

        bu_seg_p = self.conv3(bu_part_fea)
        bu_seg_h = self.conv4(bu_half_fea)
        bu_seg_f = self.conv5(bu_full_fea)
        td_seg_h = self.conv6(td_half_fea)
        td_seg_p = self.conv7(td_part_fea)
        seg_xpl = self.conv_final(xpl)

        return bu_seg_p, bu_seg_h, bu_seg_f, td_seg_h, td_seg_p, seg_xpl


class Node_DecoderModule(nn.Module):

    def __init__(self, cls_p=7, cls_h=3, cls_f=2):
        super(Node_DecoderModule, self).__init__()

        self.p_classifier = nn.ModuleList([nn.Conv2d(10, 2, kernel_size=1, padding=0, dilation=1, bias=True) for i in range(0, cls_p-1)])
        self.h_classifier = nn.ModuleList([nn.Conv2d(10, 2, kernel_size=1, padding=0, dilation=1, bias=True) for i in range(0, cls_h-1)])

    def forward(self, p_fea_list, h_fea_list):

        p_seg_list = [self.p_classifier[i](p_fea_list[i]) for i in range(0, len(p_fea_list))]
        h_seg_list = [self.h_classifier[i](h_fea_list[i]) for i in range(0, len(h_fea_list))]
        return p_seg_list, h_seg_list

class Trans_infer_rnn_tree(nn.Module):
    def __init__(self, cls_p=7, cls_h=3, cls_f=2, half_upper_nodes=[1, 2, 3, 4], half_lower_nodes=[5, 6]):
        super(Trans_infer_rnn_tree, self).__init__()
        self.half_upper_nodes = [int(i) for i in half_upper_nodes]
        self.half_lower_nodes = [int(i) for i in half_lower_nodes]
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        # self.part_leaf_list = nn.ModuleList(
        #     [leaf_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
        #      range(0, cls_p - 1)])

        self.half_upper_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.half_lower_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.full_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)

        self.td_half_upper_rnn_l = plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_lower_rnn_l = plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_upper_rnn_r = plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_lower_rnn_r = plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)

        self.td_part_leaf_list_l = nn.ModuleList(
            [plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
             range(0, cls_p - 1)])
        self.td_part_leaf_list_r = nn.ModuleList(
            [plus_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
             range(0, cls_p - 1)])


        self.p_conv = nn.Sequential(nn.Conv2d(256, 10*cls_p, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_p), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(256, 10*cls_h, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_h), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(256, 10*cls_f, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_f), nn.ReLU(inplace=False))

        self.bu_h_bg_conv = nn.Sequential(nn.Conv2d(10*cls_h, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10), nn.ReLU(inplace=False))
        self.bu_f_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_f, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))
        self.td_h_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_h, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))
        self.td_p_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_p, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))


    def forward(self, seg_p, seg_h, seg_f, p_fea, h_fea, f_fea):

        p_h_fea_list = list(torch.split(self.p_conv(p_fea), 10, dim=1))
        h_h_fea_list = list(torch.split(self.h_conv(h_fea), 10, dim=1))
        f_h_fea_list = list(torch.split(self.f_conv(f_fea), 10, dim=1))

        # half_upper [0,1,2,3], half_lower [4,5]
        bu_p_h_fea_list = p_h_fea_list
        bu_p_c_fea_list = p_h_fea_list
        # seg_p_list = torch.split(seg_p, 1, dim=1)
        half_upper_h_list = [bu_p_h_fea_list[i] for i in self.half_upper_nodes]
        half_upper_c_list = [bu_p_c_fea_list[i] for i in self.half_upper_nodes]
        half_lower_h_list = [bu_p_h_fea_list[i] for i in self.half_lower_nodes]
        half_lower_c_list = [bu_p_c_fea_list[i] for i in self.half_lower_nodes]

        half_upper_h_list.append(h_h_fea_list[1])
        half_lower_h_list.append(h_h_fea_list[2])
        # bottom-up half nodes
        half_upper_h, half_upper_c = self.half_upper_rnn(h_fea, [half_upper_h_list, half_upper_c_list])
        half_lower_h, half_lower_c = self.half_lower_rnn(h_fea, [half_lower_h_list, half_lower_c_list])
        half_bg_h = self.bu_h_bg_conv(torch.cat([h_h_fea_list[0], half_upper_h, half_lower_h], dim=1))
        bu_h_h_fea_list = [half_bg_h, half_upper_h, half_lower_h]

        # bottom-up full nodes
        full_h, full_c = self.full_rnn(f_fea, [[half_lower_h, half_lower_h, f_h_fea_list[1]], [half_upper_c, half_lower_c]])
        full_bg_h = self.bu_f_bg_conv(torch.cat([f_h_fea_list[0], full_h], dim=1))
        bu_f_h_fea_list = [full_bg_h, full_h]

        # top-down half nodes
        td_half_upper_h_l, td_half_upper_c_l = self.td_half_upper_rnn_l(h_fea,[[full_h, bu_h_h_fea_list[1]], [full_c]])
        td_half_lower_h_r, td_half_lower_c_r = self.td_half_lower_rnn_r(h_fea,[[full_h, bu_h_h_fea_list[2]], [full_c]])
        td_half_upper_h_r, td_half_upper_c_r = self.td_half_upper_rnn_r(h_fea, [[full_h, td_half_lower_h_r, bu_h_h_fea_list[1]], [full_c, td_half_lower_h_r]])
        td_half_lower_h_l, td_half_lower_c_l = self.td_half_lower_rnn_l(h_fea, [[full_h, td_half_upper_h_l, bu_h_h_fea_list[2]], [full_c, td_half_upper_c_l]])
        td_half_upper_h, td_half_lower_h = torch.max(td_half_upper_h_l,td_half_upper_h_r), torch.max(td_half_lower_h_l, td_half_lower_h_r)
        td_half_upper_c, td_half_lower_c = torch.max(td_half_upper_c_l, td_half_upper_c_r), torch.max(td_half_lower_c_l,
                                                                                                      td_half_lower_c_r)
        td_half_bg_h = self.td_h_bg_conv(torch.cat([h_h_fea_list[0], td_half_upper_h, td_half_lower_h], dim=1))
        td_h_h_fea_list = [td_half_bg_h, td_half_upper_h, td_half_lower_h]

        # top-down part nodes
        td_p_h_fea_list_l = []
        td_p_c_fea_list_l = []
        td_p_h_fea_list_r = []
        td_p_c_fea_list_r = []
        for i in range(1, self.cls_p):
            if i in self.half_upper_nodes:
                if i==self.half_upper_nodes[0]:
                    h_u_l, c_u_l = self.td_part_leaf_list_l[i -1](p_fea,
                                                           [[td_half_upper_h, bu_p_h_fea_list[i]], [td_half_upper_c]])
                else:
                    h_u_l, c_u_l = self.td_part_leaf_list_l[i - 1](p_fea,
                                                           [[td_half_upper_h, h_u_l, bu_p_h_fea_list[i]], [td_half_upper_c, c_u_l]])
                td_p_h_fea_list_l.append(h_u_l)
                td_p_c_fea_list_l.append(c_u_l)
            elif i in self.half_lower_nodes:
                if i==self.half_lower_nodes[0]:
                    h_l_l, c_l_l = self.td_part_leaf_list_l[i-1](p_fea, [[td_half_lower_h, bu_p_h_fea_list[i]], [td_half_lower_c]])
                else:
                    h_l_l, c_l_l = self.td_part_leaf_list_l[i-1](p_fea, [[td_half_lower_h, h_l_l, bu_p_h_fea_list[i]], [td_half_lower_c, c_l_l]])

                td_p_h_fea_list_l.append(h_l_l)
                td_p_c_fea_list_l.append(c_l_l)


        for i in range(1, self.cls_p):
            if self.cls_p-i in self.half_upper_nodes:
                if self.cls_p-i == self.half_upper_nodes[-1]:
                    h_u_r, c_u_r = self.td_part_leaf_list_r[self.cls_p-i-1](p_fea,
                                                                                          [[td_half_upper_h,
                                                                                            bu_p_h_fea_list[self.cls_p-i]],
                                                                                           [td_half_upper_c]])
                else:
                    h_u_r, c_u_r = self.td_part_leaf_list_r[self.cls_p-i-1](p_fea,
                                                                                          [[td_half_upper_h, h_u_r,
                                                                                            bu_p_h_fea_list[self.cls_p-i]],
                                                                                           [td_half_upper_c, c_u_r]])
                td_p_h_fea_list_r.append(h_u_r)
                td_p_c_fea_list_r.append(c_u_r)
            elif self.cls_p-i in self.half_lower_nodes:
                if self.cls_p-i == self.half_lower_nodes[-1]:
                    h_l_r, c_l_r = self.td_part_leaf_list_r[self.cls_p-i-1](p_fea, [
                        [td_half_lower_h, bu_p_h_fea_list[self.cls_p-i]], [td_half_lower_c]])
                else:
                    h_l_r, c_l_r = self.td_part_leaf_list_r[self.cls_p-i-1](p_fea, [
                        [td_half_lower_h, h_l_r, bu_p_h_fea_list[self.cls_p-i]], [td_half_lower_c, c_l_r]])

                td_p_h_fea_list_r.append(h_l_r)
                td_p_c_fea_list_r.append(c_l_r)

        td_p_h_fea_list = [bu_p_h_fea_list[0]]
        td_p_c_fea_list = [bu_p_c_fea_list[0]]

        for i in range(1,self.cls_p):
            td_p_h_fea_list.append(torch.max(td_p_h_fea_list_l[i - 1], td_p_h_fea_list_r[self.cls_p - i-1]))
            td_p_c_fea_list.append(torch.max(td_p_c_fea_list_l[i - 1], td_p_c_fea_list_r[self.cls_p - i-1]))

        p_bg_h = self.td_p_bg_conv(torch.cat(td_p_h_fea_list, dim=1))
        # td_p_h_fea_list[0] = p_bg_h
        new_td_p_h_fea_list = [p_bg_h]
        for i in range(1,len(td_p_h_fea_list)):
            new_td_p_h_fea_list.append(td_p_h_fea_list[i])

        return bu_p_h_fea_list, bu_h_h_fea_list, bu_f_h_fea_list, td_h_h_fea_list, new_td_p_h_fea_list

class Trans_infer_rnn_tree(nn.Module):
    def __init__(self, cls_p=7, cls_h=3, cls_f=2, half_upper_nodes=[1, 2, 3, 4], half_lower_nodes=[5, 6]):
        super(Trans_infer_rnn_tree, self).__init__()
        self.half_upper_nodes = [int(i) for i in half_upper_nodes]
        self.half_lower_nodes = [int(i) for i in half_lower_nodes]
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

        # self.part_leaf_list = nn.ModuleList(
        #     [leaf_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
        #      range(0, cls_p - 1)])

        self.half_upper_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.half_lower_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.full_rnn = max_tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(1, 1), bias=False)

        self.td_half_upper_rnn_l = tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_lower_rnn_l = tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_upper_rnn_r = tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.td_half_lower_rnn_r = tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False)

        self.td_part_leaf_list_l = nn.ModuleList(
            [tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
             range(0, cls_p - 1)])
        self.td_part_leaf_list_r = nn.ModuleList(
            [tree_ConvLSTMCell(input_dim=256+10, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
             range(0, cls_p - 1)])


        self.p_conv = nn.Sequential(nn.Conv2d(256, 10*cls_p, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_p), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(256, 10*cls_h, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_h), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(256, 10*cls_f, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10*cls_f), nn.ReLU(inplace=False))

        self.bu_h_bg_conv = nn.Sequential(nn.Conv2d(10*cls_h, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(10), nn.ReLU(inplace=False))
        self.bu_f_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_f, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))
        self.td_h_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_h, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))
        self.td_p_bg_conv = nn.Sequential(nn.Conv2d(10 * cls_p, 10, kernel_size=1, padding=0, stride=1, bias=False),
                                       BatchNorm2d(10), nn.ReLU(inplace=False))



    def forward(self, seg_p, seg_h, seg_f, p_fea, h_fea, f_fea):

        p_h_fea_list = list(torch.split(self.p_conv(p_fea), 10, dim=1))
        h_h_fea_list = list(torch.split(self.h_conv(h_fea), 10, dim=1))
        f_h_fea_list = list(torch.split(self.f_conv(f_fea), 10, dim=1))

        # half_upper [0,1,2,3], half_lower [4,5]
        bu_p_h_fea_list = p_h_fea_list
        bu_p_c_fea_list = p_h_fea_list
        # seg_p_list = torch.split(seg_p, 1, dim=1)
        half_upper_h_list = [bu_p_h_fea_list[i] for i in self.half_upper_nodes]
        half_upper_c_list = [bu_p_c_fea_list[i] for i in self.half_upper_nodes]
        half_lower_h_list = [bu_p_h_fea_list[i] for i in self.half_lower_nodes]
        half_lower_c_list = [bu_p_c_fea_list[i] for i in self.half_lower_nodes]

        half_upper_h_list.append(h_h_fea_list[1])
        half_lower_h_list.append(h_h_fea_list[2])
        # bottom-up half nodes
        half_upper_h, half_upper_c = self.half_upper_rnn(h_fea, [half_upper_h_list, half_upper_c_list])
        half_lower_h, half_lower_c = self.half_lower_rnn(h_fea, [half_lower_h_list, half_lower_c_list])
        half_bg_h = self.bu_h_bg_conv(torch.cat([h_h_fea_list[0], half_upper_h, half_lower_h], dim=1))
        bu_h_h_fea_list = [half_bg_h, half_upper_h, half_lower_h]
        bu_h_c_fea_list = [half_bg_h, half_upper_c, half_lower_c]

        # bottom-up full nodes
        full_h, full_c = self.full_rnn(f_fea, [[half_lower_h, half_lower_h, f_h_fea_list[1]], [half_upper_c, half_lower_c]])
        full_bg_h = self.bu_f_bg_conv(torch.cat([f_h_fea_list[0], full_h], dim=1))
        bu_f_h_fea_list = [full_bg_h, full_h]

        # top-down half nodes
        td_half_upper_h_l, td_half_upper_c_l = self.td_half_upper_rnn_l(torch.cat([h_fea, full_h], dim=1),[[bu_h_h_fea_list[1]], [bu_h_c_fea_list[1]]])
        td_half_lower_h_r, td_half_lower_c_r = self.td_half_lower_rnn_r(torch.cat([h_fea,full_h], dim=1), [[bu_h_h_fea_list[2]], [bu_h_c_fea_list[2]]])
        td_half_upper_h_r, td_half_upper_c_r = self.td_half_upper_rnn_r(torch.cat([h_fea, td_half_lower_h_r], dim=1), [[bu_h_h_fea_list[1]], [bu_h_c_fea_list[1]]])
        td_half_lower_h_l, td_half_lower_c_l = self.td_half_lower_rnn_l(torch.cat([h_fea, td_half_upper_h_l], dim =1), [[bu_h_h_fea_list[2]], [bu_h_c_fea_list[2]]])
        td_half_upper_h, td_half_lower_h = torch.max(td_half_upper_h_l,td_half_upper_h_r), torch.max(td_half_lower_h_l, td_half_lower_h_r)
        td_half_upper_c, td_half_lower_c = torch.max(td_half_upper_c_l, td_half_upper_c_r), torch.max(td_half_lower_c_l,
                                                                                                      td_half_lower_c_r)
        td_half_bg_h = self.td_h_bg_conv(torch.cat([h_h_fea_list[0], td_half_upper_h, td_half_lower_h], dim=1))
        td_h_h_fea_list = [td_half_bg_h, td_half_upper_h, td_half_lower_h]

        # top-down part nodes
        td_p_h_fea_list_l = []
        td_p_c_fea_list_l = []
        td_p_h_fea_list_r = []
        td_p_c_fea_list_r = []
        for i in range(1, self.cls_p):
            if i in self.half_upper_nodes:
                if i==self.half_upper_nodes[0]:
                    h_u_l, c_u_l = self.td_part_leaf_list_l[i -1](torch.cat([p_fea,td_half_upper_h], dim=1),
                                                           [[bu_p_h_fea_list[i]], [bu_p_c_fea_list[i]]])
                else:
                    h_u_l, c_u_l = self.td_part_leaf_list_l[i - 1](torch.cat([p_fea,h_u_l], dim=1),
                                                           [[bu_p_h_fea_list[i]], [bu_p_c_fea_list[i]]])
                td_p_h_fea_list_l.append(h_u_l)
                td_p_c_fea_list_l.append(c_u_l)
            elif i in self.half_lower_nodes:
                if i==self.half_lower_nodes[0]:
                    h_l_l, c_l_l = self.td_part_leaf_list_l[i-1](torch.cat([p_fea, td_half_lower_h],dim=1), [[bu_p_h_fea_list[i]], [bu_p_c_fea_list[i]]])
                else:
                    h_l_l, c_l_l = self.td_part_leaf_list_l[i-1](torch.cat([p_fea, h_l_l],dim=1), [[bu_p_h_fea_list[i]], [bu_p_c_fea_list[i]]])

                td_p_h_fea_list_l.append(h_l_l)
                td_p_c_fea_list_l.append(c_l_l)


        for i in range(1, self.cls_p):
            if self.cls_p-i in self.half_upper_nodes:
                if self.cls_p-i == self.half_upper_nodes[-1]:
                    h_u_r, c_u_r = self.td_part_leaf_list_r[self.cls_p-i-1](torch.cat([p_fea,td_half_upper_h],dim=1),
                                                                                          [[bu_p_h_fea_list[self.cls_p-i]],
                                                                                           [bu_p_c_fea_list[self.cls_p-i]]])
                else:
                    h_u_r, c_u_r = self.td_part_leaf_list_r[self.cls_p-i-1](torch.cat([p_fea, h_u_r],dim=1),
                                                                                          [[bu_p_h_fea_list[self.cls_p-i]],
                                                                                           [bu_p_c_fea_list[self.cls_p-i]]])
                td_p_h_fea_list_r.append(h_u_r)
                td_p_c_fea_list_r.append(c_u_r)
            elif self.cls_p-i in self.half_lower_nodes:
                if self.cls_p-i == self.half_lower_nodes[-1]:
                    h_l_r, c_l_r = self.td_part_leaf_list_r[self.cls_p-i-1](torch.cat([p_fea,td_half_lower_h], dim=1), [
                        [bu_p_h_fea_list[self.cls_p-i]], [bu_p_c_fea_list[self.cls_p-i]]])
                else:
                    h_l_r, c_l_r = self.td_part_leaf_list_r[self.cls_p-i-1](torch.cat([p_fea, h_l_r],dim=1), [
                        [bu_p_h_fea_list[self.cls_p-i]], [bu_p_c_fea_list[self.cls_p-i]]])

                td_p_h_fea_list_r.append(h_l_r)
                td_p_c_fea_list_r.append(c_l_r)

        td_p_h_fea_list = [bu_p_h_fea_list[0]]
        td_p_c_fea_list = [bu_p_c_fea_list[0]]

        for i in range(1,self.cls_p):
            td_p_h_fea_list.append(torch.max(td_p_h_fea_list_l[i - 1], td_p_h_fea_list_r[self.cls_p - i-1]))
            td_p_c_fea_list.append(torch.max(td_p_c_fea_list_l[i - 1], td_p_c_fea_list_r[self.cls_p - i-1]))

        p_bg_h = self.td_p_bg_conv(torch.cat(td_p_h_fea_list, dim=1))
        # td_p_h_fea_list[0] = p_bg_h
        new_td_p_h_fea_list = [p_bg_h]
        for i in range(1,len(td_p_h_fea_list)):
            new_td_p_h_fea_list.append(td_p_h_fea_list[i])

        return bu_p_h_fea_list, bu_h_h_fea_list, bu_f_h_fea_list, td_h_h_fea_list, new_td_p_h_fea_list

class IterTrans(nn.Module):
    def __init__(self, trans_step=2, trans_unit='rnn', fbody_cls=2, hbody_cls=3, part_cls=7):
        super(IterTrans, self).__init__()
        self.trans_step = trans_step
        self.trans_unit = trans_unit

        if self.trans_unit in ["rnn", 'lstm', 'gru']:
            self.BiRNNInfer = Trans_infer_rnn_tree(part_cls, hbody_cls, fbody_cls, half_upper_nodes=[1, 2, 3, 4], half_lower_nodes=[5, 6])

    def forward(self, x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea):
        bu_p_h_fea_list, bu_h_h_fea_list, bu_f_h_fea_list, td_h_h_fea_list, td_p_h_fea_list = self.BiRNNInfer(x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea)

        bu_part_fea = torch.cat(bu_p_h_fea_list, dim=1)
        bu_half_fea = torch.cat(bu_h_h_fea_list, dim=1)
        bu_full_fea = torch.cat(bu_f_h_fea_list, dim=1)
        td_half_fea = torch.cat(td_h_h_fea_list, dim=1)
        td_part_fea = torch.cat(td_p_h_fea_list, dim=1)
        return bu_part_fea, bu_half_fea, bu_full_fea, td_half_fea, td_part_fea

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls=3)
        self.layerf = AlphaFBDecoder(fbody_cls=2)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.iter_step = 1
        self.trans_step = 1
        self.iter_trans = IterTrans(self.trans_step, 'lstm', fbody_cls=2, hbody_cls=3, part_cls=num_classes)

        self.final = Final_DecoderModule(num_classes, cls_h=3, cls_f=2)
        self.node_seg = Node_DecoderModule(num_classes, cls_h=3, cls_f=2)

        # self.gff_p = GFF(256,70, ncls=7)



    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        p_seg0, p_fea, p_seg = self.layer6(seg, x[1], x[0])
        h_seg, h_fea = self.layerh(seg, x[1])
        f_seg, f_fea= self.layerf(seg, x[1])

        bu_part_fea, bu_half_fea, bu_full_fea, td_half_fea, td_part_fea = self.iter_trans(p_seg, h_seg, f_seg, p_fea, h_fea, f_fea)

        bu_part_seg, bu_half_seg, bu_full_seg, td_half_seg, td_part_seg, part_final = self.final(bu_part_fea, bu_half_fea, bu_full_fea, td_half_fea, td_part_fea, x[0])

        # p_seg = self.gff_p(p_fea, td_part_fea, p_seg0, part_final)
        p_seg=p_seg0+part_final

        return [p_seg, h_seg, f_seg, bu_part_seg, bu_half_seg, bu_full_seg, td_half_seg, td_part_seg, x_dsn]


class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        self.encoder = ResGridNet(block, layers)
        self.decoder = Decoder(num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(num_classes=20):
    # model = OCNet(Bottleneck, [3, 4, 6, 3], num_classes) #50
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes) #101
    # model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
