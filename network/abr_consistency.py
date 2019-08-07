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
        return output


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
        return output

class Decoder(nn.Module):
    def __init__(self, num_classes, upper_node=[1,2,3,4], lower_node=[5,6], hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()

        self.upper_node=upper_node
        self.lower_node=lower_node
        self.cls_p = num_classes
        self.cls_h = hbody_cls
        self.cls_f = fbody_cls
        self.upper_len=len(upper_node)
        self.lower_len=len(lower_node)

        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb = self.layerh(seg, x[1])
        alpha_fb = self.layerf(seg, x[1])

        _,_, h,w =x_seg.size()
        alpha_hb = F.interpolate(alpha_hb, (h,w), mode="bilinear", align_corners=True)
        alpha_fb = F.interpolate(alpha_fb, (h,w), mode="bilinear", align_corners=True)

        # x_seg = x_seg-torch.min(x_seg, dim=1, keepdim=True)[0]
        # x_seg = x_seg/(torch.max(x_seg, dim=1, keepdim=True)[0]+1e-10)
        x_seg = self.softmax(x_seg)

        # alpha_hb = alpha_hb-torch.min(alpha_hb, dim=1, keepdim=True)[0]
        # alpha_hb = alpha_hb/(torch.max(alpha_hb, dim=1, keepdim=True)[0]+1e-10)
        alpha_hb = self.softmax(alpha_hb)
        hb_list = torch.split(alpha_hb, 1, dim=1)

        # alpha_fb = alpha_fb-torch.min(alpha_fb, dim=1, keepdim=True)[0]
        # alpha_fb = alpha_fb/(torch.max(alpha_fb, dim=1, keepdim=True)[0]+1e-10)
        alpha_fb = self.softmax(alpha_fb)
        fb_list = torch.split(alpha_fb, 1, dim=1)

        p_hb_list = [hb_list[0]]
        for i in range(1, self.cls_p):
            if i in self.upper_node:
                p_hb_list.append(hb_list[0])
            else:
                p_hb_list.append(hb_list[1])

        x_seg = x_seg*torch.cat([fb_list[0]]+[fb_list[1]]*(self.cls_p-1),dim=1)*torch.cat(p_hb_list, dim=1)
        alpha_hb = alpha_hb*torch.cat([fb_list[0]]+[fb_list[1]]*(self.cls_h-1),dim=1)

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
