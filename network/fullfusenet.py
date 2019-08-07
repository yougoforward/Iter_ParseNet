import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.gating_mod import GatingBlock
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
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(24, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
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
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(24, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
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


class CombineBlock(nn.Module):
    def __init__(self, num_classes):
        super(CombineBlock, self).__init__()
        # 32 --> 24
        self.conv1 = nn.Sequential(nn.Conv2d(num_classes * 2, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, num_classes, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, alpha_x, gamma_x):
        _, _, h, w = alpha_x.size()
        gamma_x = F.interpolate(gamma_x, size=(h, w), mode='bilinear', align_corners=True)
        x_part = torch.cat([alpha_x, gamma_x], dim=1)
        output = self.conv1(x_part)
        return output


class HBCombineBlock(nn.Module):
    def __init__(self, hb_cls):
        super(HBCombineBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(hb_cls * 3, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, hb_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, alpha_x, beta_x, gamma_x):
        _, _, h, w = beta_x.size()
        alpha_x = F.interpolate(alpha_x, size=(h, w), mode='bilinear', align_corners=True)
        gamma_x = F.interpolate(gamma_x, size=(h, w), mode='bilinear', align_corners=True)
        x_hb = torch.cat([alpha_x, beta_x, gamma_x], dim=1)
        output = self.conv1(x_hb)
        return output


class FBCombineBlock(nn.Module):
    def __init__(self, fb_cls):
        super(FBCombineBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(fb_cls * 2, 24, kernel_size=1, padding=0, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1, bias=False),
                                   InPlaceABNSync(24),
                                   nn.Conv2d(24, fb_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, alpha_x, gamma_x, beta_x):
        _, _, h, w = beta_x.size()
        alpha_x = F.interpolate(alpha_x, size=(h, w), mode='bilinear', align_corners=True)
        gamma_x = F.interpolate(gamma_x, size=(h, w), mode='bilinear', align_corners=True)
        x_part = torch.cat([alpha_x, gamma_x], dim=1)
        output = self.conv1(x_part)
        return output


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
        self.fuse = CombineBlock(num_classes)
        self.fuse_hb = HBCombineBlock(3)
        self.fuse_fb = FBCombineBlock(2)

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
        x_part = self.fuse(x_seg, gama_part)
        x_hb = self.fuse_hb(alpha_hb, beta_hb, gama_hb)
        x_fb = self.fuse_fb(alpha_fb, beta_fb, beta_hb)
        gate_acts = []
        gate_acts.extend(act_bhb)
        gate_acts.extend(act_bfb)
        gate_acts.extend(act_ghb)
        gate_acts.extend(act_gp)

        # return [x_seg, alpha_hb, alpha_fb, x_dsn]

        return [x_part, x_hb, x_fb, beta_hb, beta_fb, gama_hb, gama_part, gate_acts, x_dsn]


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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(num_classes=20):
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
