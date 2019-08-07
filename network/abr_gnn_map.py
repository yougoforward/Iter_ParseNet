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
        return x_seg, x_fea


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        output = self.conv1(xfuse)
        return output, xfuse


class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_fb * skip
        output = self.conv1(xfuse)
        return output, xfuse

class GNN(nn.Module):
    def __init__(self, upper_part_node, lower_part_node):
        super(GNN, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.upper_parts = upper_part_node
        self.lower_parts = lower_part_node

    def forward(self, x_seg, alpha_hb, alpha_fb, xp_fea, xh_fea, xf_fea):
        #
        _, _, th, tw = x_seg.size()
        _, _, h, w = alpha_hb.size()
        p_list = list(torch.split(x_seg, 1, dim=1))
        h_list = list(torch.split(alpha_hb, 1, dim=1))
        f_list = list(torch.split(alpha_fb, 1, dim=1))

        upper_nodes = [p_list[i] for i in self.upper_parts]
        lower_nodes = [p_list[i] for i in self.lower_parts]

        f_att_list = torch.split(self.softmax(alpha_fb), 1, dim=1)
        h_att_list = torch.split(self.softmax(F.interpolate(alpha_hb, (th, tw), mode="bilinear", align_corners=True)), 1, dim=1)

        # full graph
        f_0_new = (f_list[1]+h_list[0])/2.0

        comp_hf = torch.max(torch.cat([h_list[1], h_list[2]], dim=1), dim=1, keepdim=True)[0]
        f_1_new = (f_list[1]+comp_hf)/2.0
        f_list_new = [f_0_new, f_1_new]

        # half graph
        h_0_new = (h_list[0]+h_list[0]*f_att_list[0]+F.interpolate(p_list[0], (h,w), mode="bilinear", align_corners=True))/3.0

        comp_phu = torch.max(torch.cat(upper_nodes, dim=1), dim=1, keepdim=True)[0]
        comp_phu = F.interpolate(comp_phu, (h,w), mode="bilinear", align_corners=True)
        h_1_new = (h_list[1]+h_list[1]*f_att_list[1]+comp_phu)/3.0

        comp_phl = torch.max(torch.cat(lower_nodes, dim=1), dim=1, keepdim=True)[0]
        comp_phl = F.interpolate(comp_phl, (h,w), mode="bilinear", align_corners=True)
        h_2_new = (h_list[2]+h_list[2]*f_att_list[1]+comp_phl)/3.0
        h_list_new = [h_0_new, h_1_new, h_2_new]

        # part graph
        p_list_new=[]
        for i in range(len(p_list)):
            if i==0:
                p_new = (p_list[0]+p_list[0]*h_att_list[0])/2.0
            elif i in self.upper_parts:
                p_new = (p_list[i]+p_list[i]*h_att_list[1])/2.0
            elif i in self.lower_parts:
                p_new = (p_list[i]+p_list[i]*h_att_list[2])/2.0

            p_list_new.append(p_new)

        f_seg = torch.cat(f_list_new, dim=1)
        h_seg = torch.cat(h_list_new, dim=1)
        p_seg = torch.cat(p_list_new, dim=1)

        return p_seg, h_seg, f_seg

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
        self.gnn = GNN(upper_part_node=[1,2,3,4], lower_part_node=[5,6])

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, xp_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, xh_fea = self.layerh(seg, x[1])
        alpha_fb, xf_fea = self.layerf(seg, x[1])

        x_seg, alpha_hb, alpha_fb = self.gnn(x_seg,alpha_hb,alpha_fb, xp_fea, xh_fea, xf_fea)

        return [x_seg, alpha_hb, alpha_fb, x_dsn]


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
