import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, SEModule
from modules.gating_mod import GatingBlock
from modules.se_mod import MagicModule

from . import convolutional_rnn


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

        self.conv3 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False),
                                   nn.Conv2d(64, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, hb_fea):
        x_fea = self.conv1(x)
        hb_fea = self.conv2(hb_fea)
        x_fuse = torch.cat([x_fea, hb_fea], dim=1)
        x_seg = self.conv3(x_fuse)
        return x_seg


class GamaPartDecoder(nn.Module):
    def __init__(self, hbody_cls, num_classes):
        super(GamaPartDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(hbody_cls, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False),
                                   nn.Conv2d(64, num_classes, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, x, x_fea):
        x = self.conv1(x)
        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)
        x_seg = self.conv3(x_fuse)
        return x_seg

class Trans_infer(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(Trans_infer, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(num_classes1+num_classes2, 32, kernel_size=1, padding=0, stride=1, bias=False),
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


class Trans_infer_rnn_fea(nn.Module):
    def __init__(self, num_classes1, num_classes2, rnn_type='RNN'):
        super(Trans_infer_rnn_fea, self).__init__()
        self.ts=2
        self.rnn_type = rnn_type
        if self.rnn_type == 'rnn':
            self.net = convolutional_rnn.Conv2dRNN(in_channels=96, out_channels=256,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=False,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type == 'gru':
            self.net = convolutional_rnn.Conv2dGRU(in_channels=96, out_channels=256,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=False,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type == 'lstm':
            self.net = convolutional_rnn.Conv2dLSTM(in_channels=96, out_channels=256,
                                                    kernel_size=1,
                                                    num_layers=1,
                                                    bias=True,
                                                    batch_first=False,
                                                    dropout=0.,
                                                    bidirectional=False,
                                                    stride=1,
                                                    dilation=1,
                                                    groups=1)

        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(64), nn.ReLU(inplace=False))

        self.conv1_1 = nn.Sequential(nn.Conv2d(num_classes1+num_classes2, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False),
                                   )

        self.conv3 = nn.Conv2d(256, num_classes2, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, x_coarse, x_fea, x_coarse_fea):

        x=torch.cat([x,x_coarse], dim=1)
        x=self.conv1_1(x)

        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)

        _, x_fine_fea = self.net(x_fuse.unsqueeze(0).repeat(self.ts,1,1,1,1), x_coarse_fea.unsqueeze(0))
        x_fine = self.conv3(x_fine_fea[0])

        return x_fine, x_fine_fea[0]

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

            self.PartHalfInfer = Trans_infer_rnn_fea(part_cls, hbody_cls, self.trans_unit)
            self.HalfFullInfer = Trans_infer_rnn_fea(hbody_cls, fbody_cls, self.trans_unit)
            self.FullHalfInfer = Trans_infer_rnn_fea(fbody_cls, hbody_cls, self.trans_unit)
            self.HalfPartInfer = Trans_infer_rnn_fea(hbody_cls, part_cls, self.trans_unit)


    def forward(self, x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea):
        x_hbody, h_fea = self.PartHalfInfer(x_part, x_hbody, p_fea, h_fea)
        x_fbody, f_fea = self.HalfFullInfer(x_hbody, x_fbody, h_fea, f_fea)
        x_hbody, h_fea = self.FullHalfInfer(x_fbody, x_hbody, f_fea, h_fea)
        x_part , p_fea= self.HalfPartInfer(x_hbody, x_part, h_fea, p_fea)

        return x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea

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
        self.iter_step = 2
        self.trans_step = 2
        self.iter_trans = IterTrans(self.trans_step, 'rnn', part_cls=num_classes)

        self.fuse = CombineBlock(num_classes)
    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, hb_fea = self.layerh(seg, x[1])
        alpha_fb, fb_fea = self.layerf(seg, x[1])
        _, _, h, w = x_fea.size()
        x_seg_d = F.interpolate(x_seg, size=(h, w), mode='bilinear', align_corners=True)
        x_seg1, alpha_hb1, alpha_fb1, x_fea1, hb_fea1, fb_fea1 = self.iter_trans(x_seg_d, alpha_hb, alpha_fb, x_fea, hb_fea, fb_fea)
        x_seg2, alpha_hb2, alpha_fb2, x_fea2, hb_fea2, fb_fea2 = self.iter_trans(x_seg1, alpha_hb1, alpha_fb1, x_fea1, hb_fea1,
                                                                           fb_fea1)
        x_seg_final = self.fuse(x_seg, x_seg2)
        return [x_seg, alpha_hb, alpha_fb, x_seg1, alpha_hb1, alpha_fb1, x_seg2, alpha_hb2, alpha_fb2, x_seg_final, x_dsn]


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
