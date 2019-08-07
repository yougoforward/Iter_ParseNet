import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, SEModule, GatingBlock
from modules.se_mod import MagicModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
affine_par = True


class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
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


class ResGridNet(nn.Module):
    """The dilation rates of the last res-block are multi-grid."""

    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResGridNet, self).__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, affine=True)
        self.relu3 = nn.ReLU(inplace=False)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))

        self.inplanes = planes * block.expansion
        if multi_grid:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation ** (i+1)))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x2, x3, x4, x5]


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))
        self.conv2 = nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        hb_fea = self.conv1(xfuse)
        hb_seg = self.conv2(hb_fea)
        return hb_seg, hb_fea


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
        fb_fea = self.conv1(xfuse)
        fb_seg = self.conv2(fb_fea)
        return fb_seg, fb_fea


class BetaHBDecoder(nn.Module):
    def __init__(self, num_classes, hbody_cls):
        super(BetaHBDecoder, self).__init__()
        self.gate = GatingBlock(in_dim=num_classes, out_dim=num_classes, force_hard=True)
        self.conv1 = nn.Sequential(nn.Conv2d(num_classes, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(24), nn.ReLU(inplace=False),
                                   nn.Conv2d(24, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x):
        x, act_bhb = self.gate(x)
        output = self.conv1(x)
        return output, act_bhb


class BetaFBDecoder(nn.Module):
    def __init__(self, hbody_cls, fbody_cls):
        super(BetaFBDecoder, self).__init__()
        self.gate = GatingBlock(in_dim=hbody_cls, out_dim=hbody_cls, force_hard=True)
        self.conv1 = nn.Sequential(nn.Conv2d(hbody_cls, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(24), nn.ReLU(inplace=False),
                                   nn.Conv2d(24, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x):
        x, act_bfb = self.gate(x)
        output = self.conv1(x)
        return output, act_bfb


class GamaHBDecoder(nn.Module):
    def __init__(self, fbody_cls, hbody_cls):
        super(GamaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(fbody_cls, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False))

        self.gate = GatingBlock(in_dim=96, out_dim=96, force_hard=True)
        self.conv3 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False),
                                   nn.Conv2d(64, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, hb_fea):
        x_fea = self.conv1(x)
        hb_fea = self.conv2(hb_fea)
        x_fuse = torch.cat([x_fea, hb_fea], dim=1)
        x_fuse, act_ghb = self.gate(x_fuse)
        x_seg = self.conv3(x_fuse)
        return x_seg, act_ghb


class GamaPartDecoder(nn.Module):
    def __init__(self, hbody_cls, num_classes):
        super(GamaPartDecoder, self).__init__()
        self.gate = GatingBlock(in_dim=hbody_cls, out_dim=hbody_cls, force_hard=True)
        self.conv1 = nn.Sequential(nn.Conv2d(hbody_cls, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False))

        self.gate = GatingBlock(in_dim=96, out_dim=96, force_hard=True)
        self.conv3 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False),
                                   nn.Conv2d(64, num_classes, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, x_fea):
        x = self.conv1(x)
        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)
        x_fuse, act_gp = self.gate(x_fuse)
        x_seg = self.conv3(x_fuse)
        return x_seg, act_gp


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls=3)
        self.layerf = AlphaFBDecoder(fbody_cls=2)
        self.layerbh = BetaHBDecoder(num_classes=num_classes, hbody_cls=3)
        self.layerbf = BetaFBDecoder(hbody_cls=3, fbody_cls=2)
        self.layergh = GamaHBDecoder(fbody_cls=2, hbody_cls=3)
        self.layergp = GamaPartDecoder(hbody_cls=3, num_classes=num_classes)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, hb_fea = self.layerh(seg, x[1])
        alpha_fb, fb_fea = self.layerf(seg, x[1])
        beta_hb, act_bhb = self.layerbh(x_seg)
        beta_fb, act_bfb = self.layerbf(alpha_hb)
        gama_hb, act_ghb = self.layergh(alpha_fb, hb_fea)
        gama_part, act_gp = self.layergp(alpha_hb, x_fea)
        gate_acts = []
        gate_acts.extend(act_bhb)
        gate_acts.extend(act_bfb)
        gate_acts.extend(act_ghb)
        gate_acts.extend(act_gp)

        return [x_seg, alpha_hb, alpha_fb, beta_hb, beta_fb, gama_hb, gama_part, gate_acts, x_dsn]


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

        nn.init.constant_(self.decoder.layer5.atte_branch[2].weights.weight, 0)
        nn.init.constant_(self.decoder.layer5.atte_branch[2].weights.bias, 0)
        # initialize the bias of the last fc for initial opening rate of the gate of about 85%
        # nn.init.constant_(self.decoder.layerbh.gate.fc2.bias.data[0], 0.1)
        # nn.init.constant_(self.decoder.layerbh.gate.fc2.bias.data[1], 2)
        # nn.init.constant_(self.decoder.layerbf.gate.fc2.bias.data[0], 0.1)
        # nn.init.constant_(self.decoder.layerbf.gate.fc2.bias.data[1], 2)
        # nn.init.constant_(self.decoder.layergh.gate.fc2.bias.data[0], 0.1)
        # nn.init.constant_(self.decoder.layergh.gate.fc2.bias.data[1], 2)
        # nn.init.constant_(self.decoder.layergp.gate.fc2.bias.data[0], 0.1)
        # nn.init.constant_(self.decoder.layergp.gate.fc2.bias.data[1], 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(num_classes=20):
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
