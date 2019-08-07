import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule

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
        return x_seg, xt_fea
        # return xt_fea

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
        # return xfuse

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
        # return xfuse

class GNN(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(GNN, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim

    def forward(self, xp, xh, xf):

        return xp, xh, xf

class GNN_infer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim

        # feature transform
        self.p_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * cls_p, kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * cls_p), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * cls_h, kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * cls_h), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * cls_f, kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * cls_f), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn=GNN(self.ch_in, self.hidden, self.cp, self.ch, self.cf)

        # feature d_transform
        self.p_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_p, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim), nn.ReLU(inplace=False))
        self.h_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_h, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim), nn.ReLU(inplace=False))
        self.f_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_f, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim), nn.ReLU(inplace=False))

        # # classifier
        # self.p_cls = nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True)
        # self.h_cls = nn.Conv2d(in_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True)
        # self.f_cls = nn.Conv2d(in_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, xp, xh, xf):
        # feature transform
        p_h_fea_list = list(torch.split(self.p_conv(xp), self.hidden, dim=1))
        h_h_fea_list = list(torch.split(self.h_conv(xh), self.hidden, dim=1))
        f_h_fea_list = list(torch.split(self.f_conv(xf), self.hidden, dim=1))

        # gnn infer
        # p_h_fea_list, h_h_fea_list, f_h_fea_list = self.gnn(p_h_fea_list, h_h_fea_list, f_h_fea_list)
        # p_h_fea_list, h_h_fea_list, f_h_fea_list = self.gnn(p_h_fea_list, h_h_fea_list, f_h_fea_list, xp, xh, xf)

        #feature d_transform
        xp_infer = self.p_dconv(torch.cat(p_h_fea_list, dim=1))
        xh_infer = self.h_dconv(torch.cat(h_h_fea_list, dim=1))
        xf_infer = self.f_dconv(torch.cat(f_h_fea_list, dim=1))

        return xp_infer, xh_infer, xf_infer

        # #classifier
        # xp_seg = self.p_cls(xp_infer)
        # xh_seg = self.h_cls(xh_infer)
        # xf_seg = self.f_cls(xf_infer)

        # return xp_seg, xh_seg, xf_seg

class Final_classifer(nn.Module):
    def __init__(self, in_dim=256, cls_p=7, cls_h=3, cls_f=2):
        super(Final_classifer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        # classifier
        self.p_cls = nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True)
        self.h_cls = nn.Conv2d(in_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True)
        self.f_cls = nn.Conv2d(in_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True)

        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(in_dim+48, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False))

        # self.conv4 = nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xp, xh, xf, xl):
        #classifier
        _, _, th, tw = xl.size()
        xt = F.interpolate(xp, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        xp_seg = self.p_cls(x_fea)

        # xp_seg = self.p_cls(xp)
        xh_seg = self.h_cls(xh)
        xf_seg = self.f_cls(xf)

        return xp_seg, xh_seg, xf_seg

class CombineBlock(nn.Module):
    def __init__(self, nclasses, num_branches):
        super(CombineBlock, self).__init__()
        # 32 --> 24
        self.conv1 = nn.Sequential(nn.Conv2d(nclasses*num_branches, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, nclasses, kernel_size=1, padding=0, stride=1, bias=True))


    def forward(self, inputs):

        # inputs = []
        _, _, h, w = inputs[0].size()
        outputs = [inputs[0]]
        for ind, x in enumerate(inputs):
            if ind>0:
                up_x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
                outputs.append(up_x)

        output = torch.cat(outputs, dim=1)
        output = self.conv1(output)
        return output

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)

        self.fuse_p = CombineBlock(num_classes, 2)
        self.fuse_h = CombineBlock(hbody_cls, 2)
        self.fuse_f = CombineBlock(fbody_cls, 2)


        self.gnn_infer=GNN_infer(in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2)

        self.classifier = Final_classifer(in_dim=256, cls_p=7, cls_h=3, cls_f=2)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, alpha_hb_fea= self.layerh(seg, x[1])
        alpha_fb, alpha_fb_fea = self.layerf(seg, x[1])
        # x_fea = self.layer6(seg, x[1], x[0])
        # alpha_hb_fea= self.layerh(seg, x[1])
        # alpha_fb_fea = self.layerf(seg, x[1])

        xp_seg, xh_seg, xf_seg=self.gnn_infer(x_fea,alpha_hb_fea,alpha_fb_fea)
        p_seg, h_seg, f_seg=self.classifier(xp_seg+x_fea, xh_seg+alpha_hb_fea, xf_seg+alpha_fb_fea, x[0])

        # p_seg = self.fuse_p([x_seg, p_seg])
        # h_seg = self.fuse_h([alpha_hb, h_seg])
        # f_seg = self.fuse_f([alpha_fb, f_seg])
        p_seg = x_seg+p_seg
        h_seg = alpha_hb+h_seg
        f_seg = alpha_fb+f_seg
        return [p_seg, h_seg, f_seg, x_dsn]



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
