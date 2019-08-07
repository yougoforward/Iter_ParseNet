import functools

import torch
import torch.nn as nn
from torch.nn import functional as F


from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from . import convolutional_rnn
from modules.convlstm import *
from inplace_abn.bn import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

# class DecoderModule(nn.Module):
#
#     def __init__(self, num_classes):
#         super(DecoderModule, self).__init__()
#         self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
#                                    BatchNorm2d(512), nn.ReLU(inplace=False))
#         self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))
#
#         self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(48), nn.ReLU(inplace=False))
#
#         self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False),
#                                    nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))
#
#         self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
#         self.alpha = nn.Parameter(torch.ones(1))
#
#     def forward(self, xt, xm, xl):
#         _, _, h, w = xm.size()
#         xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
#         _, _, th, tw = xl.size()
#         xt_fea = self.conv1(xt)
#
#
#         xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
#         xl = self.conv2(xl)
#         x = torch.cat([xt, xl], dim=1)
#         x_fea = self.conv3(x)
#         x_seg = self.conv4(x_fea)
#         return x_seg, xt_fea

class Final_DecoderModule(nn.Module):

    def __init__(self, cls_p=7, cls_h=3, cls_f=2):
        super(Final_DecoderModule, self).__init__()
        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv4 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv5 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv6 = nn.Conv2d(256, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv7 = nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv8 = nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xp, xh, xf, xl):

        _, _, th, tw = xl.size()

        xp = F.interpolate(xp, size=(th, tw), mode='bilinear', align_corners=True)
        xh = F.interpolate(xh, size=(th, tw), mode='bilinear', align_corners=True)
        xf = F.interpolate(xf, size=(th, tw), mode='bilinear', align_corners=True)

        xl = self.conv2(xl)

        xpl = torch.cat([xp, xl], dim=1)
        xhl = torch.cat([xh, xl], dim=1)
        xfl = torch.cat([xf, xl], dim=1)

        xpl=self.conv3(xpl)
        xhl=self.conv4(xhl)
        xfl=self.conv5(xfl)

        seg_p=self.conv6(xpl)
        seg_h=self.conv7(xhl)
        seg_f=self.conv8(xfl)

        return seg_p, seg_h, seg_f


class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16)
                                   )

        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm):
        _, _, h, w = xm.size()
        xup = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True)
        xt = xup + self.alpha * xm

        xt_fea = self.conv1(xt)
        x_seg = self.conv2(xt_fea)

        return x_seg, xt_fea

class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16)
                                   )
        self.conv2 = nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, dilation=1, bias=True)


        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        xfuse = self.conv1(xfuse)
        x_seg = self.conv2(xfuse)

        return x_seg, xfuse


class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16)
                                   )
        self.conv2 = nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, dilation=1, bias=True)


        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_fb * skip
        xfuse = self.conv1(xfuse)
        x_seg = self.conv2(xfuse)

        return x_seg, xfuse

class Trans_infer(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(Trans_infer, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(num_classes1, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False),
                                   nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False),
                                   nn.Conv2d(64, num_classes2, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, x_coarse, x_fea):
        x = torch.cat([x, x_coarse], dim=1)
        x = self.conv1(x)

        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)
        x_seg = self.conv3(x_fuse)
        return x_seg

class Trans_infer_rnn(nn.Module):
    def __init__(self, cls_p=7, cls_h=3, cls_f=2, rnn_type='RNN'):
        super(Trans_infer_rnn, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'rnn':
            self.net = convolutional_rnn.Conv2dRNN(in_channels=128, out_channels=128,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=True,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type == 'gru':
            self.net = convolutional_rnn.Conv2dGRU(in_channels=128, out_channels=128,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=True,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type == 'lstm':
            self.net = convolutional_rnn.Conv2dLSTM(in_channels=128, out_channels=128,
                                                    kernel_size=1,
                                                    num_layers=1,
                                                    bias=True,
                                                    batch_first=False,
                                                    dropout=0.,
                                                    bidirectional=True,
                                                    stride=1,
                                                    dilation=1,
                                                    groups=1)

        self.p_fuse = nn.Sequential(nn.Conv2d(256+cls_p, 128, kernel_size=1, padding=0, dilation=1, bias=False),
                                    BatchNorm2d(128), nn.ReLU(inplace=False))
        self.h_fuse = nn.Sequential(nn.Conv2d(256+cls_h, 128, kernel_size=1, padding=0, dilation=1, bias=False),
                                    BatchNorm2d(128), nn.ReLU(inplace=False))
        self.f_fuse = nn.Sequential(nn.Conv2d(256+cls_f, 128, kernel_size=1, padding=0, dilation=1, bias=False),
                                    BatchNorm2d(128), nn.ReLU(inplace=False))

    def forward(self, seg_p,seg_h, seg_f, p_fea, h_fea, f_fea):
        p_fea = self.p_fuse(torch.cat([p_fea,seg_p] , dim=1))
        h_fea = self.h_fuse(torch.cat([h_fea,seg_h] , dim=1))
        f_fea = self.f_fuse(torch.cat([f_fea,seg_f] , dim=1))


        x_fuse = torch.cat([p_fea.unsqueeze(0), h_fea.unsqueeze(0), f_fea.unsqueeze(0)], dim=0)

        x_fuse_refine, h = self.net(x_fuse)
        p_fea, h_fea, f_fea = torch.split(x_fuse_refine, 1, dim=0)
        p_fea = p_fea.squeeze(0)
        h_fea = h_fea.squeeze(0)
        f_fea = f_fea.squeeze(0)
        return p_fea, h_fea, f_fea


class Trans_infer_rnn_tree(nn.Module):
    def __init__(self, cls_p=7, cls_h=3, cls_f=2, half_upper_nodes=[1,2,3,4], half_lower_nodes=[5,6]):
        super(Trans_infer_rnn_tree, self).__init__()
        self.half_upper_nodes= half_upper_nodes
        self.half_lower_nodes= half_lower_nodes
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.part_leaf_list = nn.ModuleList([leaf_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(3, 3), bias=False) for i in range(0, cls_p-1)])

        self.half_upper_rnn = tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(3, 3), bias=False)
        self.half_lower_rnn = tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(3, 3), bias=False)

        self.full_rnn =tree_ConvLSTMCell(input_dim=256, hidden_dim=10, kernel_size=(3, 3), bias=False)


    def forward(self, seg_p, seg_h, seg_f, p_fea, h_fea, f_fea):

        #half_upper [0,1,2,3], half_lower [4,5]
        half_h_list=[]
        half_c_list=[]
        for i in range(0, self.cls_p-1):
            h,c=self.part_leaf_list[i](p_fea)
            half_h_list.append(h)
            half_c_list.append(c)

        half_upper_h_list = [half_h_list[i - 1] for i in self.half_upper_nodes]
        half_upper_c_list = [half_c_list[i - 1] for i in self.half_upper_nodes]
        half_lower_h_list = [half_h_list[i - 1] for i in self.half_lower_nodes]
        half_lower_c_list = [half_c_list[i - 1] for i in self.half_lower_nodes]

        half_upper_h, half_upper_c = self.half_upper_rnn(h_fea, [half_upper_h_list,half_upper_c_list])
        half_lower_h, half_lower_c = self.half_lower_rnn(h_fea, [half_lower_h_list,half_lower_c_list])

        full_h, full_c = self.full_rnn(f_fea, [[half_lower_h, half_lower_h], [half_upper_c, half_lower_c]])

        return half_h_list,  [half_lower_h, half_lower_h], full_h, half_c_list, [half_upper_c, half_lower_c], full_c


class IterTrans(nn.Module):
    def __init__(self, trans_step=2, trans_unit='rnn', fbody_cls=2, hbody_cls=3, part_cls=7):
        super(IterTrans, self).__init__()
        self.trans_step = trans_step
        self.trans_unit = trans_unit


        if self.trans_unit == "conv":
            self.PartHalfInfer = Trans_infer(part_cls, hbody_cls)
            self.HalfFullInfer = Trans_infer(hbody_cls, fbody_cls)
            self.FullHalfInfer = Trans_infer(fbody_cls, hbody_cls)
            self.HalfPartInfer = Trans_infer(hbody_cls, part_cls)
        if self.trans_unit in ["rnn", 'lstm', 'gru']:
            self.BiRNNInfer = Trans_infer_rnn(part_cls, hbody_cls, fbody_cls, self.trans_unit)

    def forward(self, x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea):
        p_fea, h_fea, f_fea = self.BiRNNInfer(x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea)

        return p_fea, h_fea, f_fea

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)

        self.conv_skip = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))

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

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x[1]= self.conv_skip(x[1])
        x_seg, x_fea = self.layer6(seg, x[1])
        hb_seg, hb_fea = self.layerh(seg, x[1])
        fb_seg, fb_fea = self.layerf(seg, x[1])

        x_fea1, hb_fea1, fb_fea1 = self.iter_trans(x_seg, hb_seg, fb_seg, x_fea, hb_fea, fb_fea)


        x_seg1, hb_seg1, fb_seg1 = self.final(x_fea1, hb_fea1, fb_fea1, x[0])
        return [x_seg1, hb_seg1, fb_seg1, x_seg, hb_seg, fb_seg, x_dsn]


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

    return model
