import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from modules.seg_hrnet import get_seg_model
from modules.jpu_mod import hr_JPU

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class DecoderModule(nn.Module):

    def __init__(self, in_dim, num_classes):
        super(DecoderModule, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))


    def forward(self, x):

        output = self.conv1(x)
        return output


class AlphaHBDecoder(nn.Module):
    def __init__(self, in_dim, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x):
        output = self.conv1(x)
        return output


class AlphaFBDecoder(nn.Module):
    def __init__(self, in_dim, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x):
        output = self.conv1(x)
        return output
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(512, 256, 2)
        self.layer6 = DecoderModule(512, num_classes)
        self.layerh = AlphaHBDecoder(512, hbody_cls=3)
        self.layerf = AlphaFBDecoder(512, fbody_cls=2)

        self.layer_dsn = nn.Sequential(nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.jpu = hr_JPU([32, 64, 128, 256], width=128, norm_layer=BatchNorm2d)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x=list(x)
        # # Upsampling
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        # x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        # x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')
        #
        # x_cat = torch.cat([x[0], x1, x2, x3], 1)
        x_cat = self.jpu(x)
        x_pool = self.pool(x_cat)

        x_context = self.layer5(x_pool)
        _, _, h, w = x[0].size()
        x_context = F.interpolate(x_context, (h, w), mode='bilinear', align_corners=True)
        x_cat = x_cat+x_context

        x_seg = self.layer6(x_cat)
        alpha_hb = self.layerh(x_cat)
        alpha_fb = self.layerf(x_cat)

        return [x_seg, alpha_hb, alpha_fb]



class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        # self.encoder = ResGridNet(block, layers)
        self.encoder = get_seg_model()
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
    model = OCNet(Bottleneck, [3, 4, 6, 3], num_classes) #50
    # model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes) #101

    return model
