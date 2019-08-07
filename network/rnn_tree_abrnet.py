import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from . import convolutional_rnn
from modules.convlstm import *

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

    def forward(self, x1, x2, seg1, seg2):

        x1_se = self.conv1(x1)
        x2_se = self.conv2(x2)
        seg = seg1*(1+x1_se) + (1-x1_se)*(seg2*x2_se)
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
        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(364, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        # self.conv4 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False),
        #                            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False))
        # self.conv5 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False),
        #                            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv6 = nn.Conv2d(256, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv7 = nn.Conv2d(276, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv8 = nn.Conv2d(266, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xp, xh, xf, xl):
        _, _, th, tw = xl.size()

        xp = F.interpolate(xp, size=(th, tw), mode='bilinear', align_corners=True)
        # xh = F.interpolate(xh, size=(th, tw), mode='bilinear', align_corners=True)
        # xf = F.interpolate(xf, size=(th, tw), mode='bilinear', align_corners=True)

        xl = self.conv2(xl)

        xpl = torch.cat([xp, xl], dim=1)
        # xhl = torch.cat([xh, xl], dim=1)
        # xfl = torch.cat([xf, xl], dim=1)

        xpl = self.conv3(xpl)
        # xhl = self.conv4(xhl)
        # xfl = self.conv5(xfl)

        seg_p = self.conv6(xpl)
        seg_h = self.conv7(xh)
        seg_f = self.conv8(xf)

        return seg_p, seg_h, seg_f


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

        # self.p_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=False),
        #                            BatchNorm2d(128), nn.ReLU(inplace=False))
        # self.h_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=False),
        #                            BatchNorm2d(128), nn.ReLU(inplace=False))
        # self.f_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=False),
        #                            BatchNorm2d(128), nn.ReLU(inplace=False))

        self.part_leaf_list = nn.ModuleList(
            [leaf_ConvLSTMCell(input_dim=256+1, hidden_dim=10, kernel_size=(1, 1), bias=False) for i in
             range(0, cls_p - 1)])

        self.half_upper_rnn = tree_ConvLSTMCell(input_dim=256+1, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.half_lower_rnn = tree_ConvLSTMCell(input_dim=256+1, hidden_dim=10, kernel_size=(1, 1), bias=False)
        self.full_rnn = tree_ConvLSTMCell(input_dim=256+1, hidden_dim=10, kernel_size=(1, 1), bias=False)




    def forward(self, seg_p, seg_h, seg_f, p_fea, h_fea, f_fea):

        # p_fea=self.p_conv(p_fea)
        # h_fea=self.h_conv(h_fea)
        # f_fea=self.f_conv(f_fea)

        # half_upper [0,1,2,3], half_lower [4,5]
        half_h_list = []
        half_c_list = []
        seg_p_list = torch.split(seg_p, 1, dim=1)
        for i in range(0, self.cls_p - 1):
            h, c = self.part_leaf_list[i](torch.cat([p_fea, seg_p_list[i+1]], dim=1))
            half_h_list.append(h)
            half_c_list.append(c)

        half_upper_h_list = [half_h_list[i - 1] for i in self.half_upper_nodes]
        half_upper_c_list = [half_c_list[i - 1] for i in self.half_upper_nodes]
        half_lower_h_list = [half_h_list[i - 1] for i in self.half_lower_nodes]
        half_lower_c_list = [half_c_list[i - 1] for i in self.half_lower_nodes]

        seg_h_list = torch.split(seg_h, 1, dim=1)

        half_upper_h, half_upper_c = self.half_upper_rnn(torch.cat([h_fea,seg_h_list[1]], dim=1), [half_upper_h_list, half_upper_c_list])
        half_lower_h, half_lower_c = self.half_lower_rnn(torch.cat([h_fea,seg_h_list[2]], dim=1), [half_lower_h_list, half_lower_c_list])

        seg_f_list = torch.split(seg_f, 1, dim=1)
        full_h, full_c = self.full_rnn(torch.cat([f_fea, seg_f_list[1]], dim=1) , [[half_lower_h, half_lower_h], [half_upper_c, half_lower_c]])

        return half_h_list, [half_lower_h, half_lower_h], full_h, half_c_list, [half_upper_c, half_lower_c], full_c


class IterTrans(nn.Module):
    def __init__(self, trans_step=2, trans_unit='rnn', fbody_cls=2, hbody_cls=3, part_cls=7):
        super(IterTrans, self).__init__()
        self.trans_step = trans_step
        self.trans_unit = trans_unit

        if self.trans_unit in ["rnn", 'lstm', 'gru']:
            self.BiRNNInfer = Trans_infer_rnn_tree(part_cls, hbody_cls, fbody_cls, half_upper_nodes=[1, 2, 3, 4], half_lower_nodes=[5, 6])

    def forward(self, x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea):
        part_h_list, half_h_list, full_h, part_c_list, part_c_list, full_c = self.BiRNNInfer(x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea)

        part_fea = torch.cat([torch.cat(part_h_list, dim=1), p_fea], dim=1)
        half_fea = torch.cat([torch.cat(half_h_list, dim=1), h_fea], dim=1)
        full_fea = torch.cat([full_h,f_fea], dim=1)

        return part_fea, half_fea, full_fea, part_h_list, half_h_list

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

        self.gff_p = GFF(256,316, ncls=7)
        self.gff_h = GFF(256,276, ncls=3)
        self.gff_f = GFF(256,266, ncls=2)


    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        p_seg0, p_fea, p_seg = self.layer6(seg, x[1], x[0])
        h_seg, h_fea = self.layerh(seg, x[1])
        f_seg, f_fea= self.layerf(seg, x[1])

        p_fea1, h_fea1, f_fea1, p_fea_list, h_fea_list = self.iter_trans(p_seg, h_seg, f_seg, p_fea, h_fea, f_fea)

        p_seg1, h_seg1, f_seg1 = self.final(p_fea1, h_fea1, f_fea1, x[0])
        p_seg_list, h_seg_list = self.node_seg(p_fea_list, h_fea_list)

        p_seg = self.gff_p(p_fea, p_fea1, p_seg0, p_seg1)
        h_seg = self.gff_h(h_fea, h_fea1, h_seg, h_seg1)
        f_seg = self.gff_f(f_fea, f_fea1, f_seg, f_seg1)

        # return [p_seg1, h_seg1, f_seg1, p_seg, h_seg, f_seg, p_seg_list, h_seg_list, x_dsn]
        # return [p_seg1, h_seg1, f_seg1, p_seg, h_seg, f_seg, p_seg_list, h_seg_list, x_dsn]

        return [p_seg, h_seg, f_seg, p_seg_list, h_seg_list, x_dsn]


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
    # model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes) #101
    model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
