import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, SEModule
from modules.se_mod import SEOCModule
from . import convolutional_rnn

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
affine_par = True


class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv0_skip = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        # self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
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
        xm = self.conv0_skip(xm)
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        _, _, th, tw = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, affine=affine_par)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, affine=affine_par)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, affine=affine_par)
        self.relu3 = nn.ReLU(inplace=False)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation))
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


class HBodyDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(HBodyDecoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   )
        self.project=nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        skip = self.conv0(skip)
        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = torch.cat([xup, skip], dim=1)

        xfuse = self.conv1(xfuse)
        output = self.project(xfuse)
        return output, xfuse


class FBodyDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(FBodyDecoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   )
        self.project=nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x, skip):
        _, _, h, w = skip.size()

        skip = self.conv0(skip)
        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = torch.cat([xup, skip], dim=1)

        xfuse = self.conv1(xfuse)
        output = self.project(xfuse)
        return output, xfuse

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


class Trans_infer_rnn_ed(nn.Module):
    def __init__(self, num_classes1, num_classes2, rnn_type='RNN'):
        super(Trans_infer_rnn_ed, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'rnn':
            self.net = convolutional_rnn.Conv2dRNN(in_channels=96, out_channels=32,
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
            self.net = convolutional_rnn.Conv2dGRU(in_channels=96, out_channels=32,
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
            self.net = convolutional_rnn.Conv2dLSTM(in_channels=96, out_channels=32,
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

        self.conv1_1 = nn.Sequential(nn.Conv2d(num_classes1, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False),
                                   nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False)
                                   )
        self.conv1_2 = nn.Sequential(nn.Conv2d(num_classes2, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False),
                                   nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False)
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(32), nn.ReLU(inplace=False),
                                   nn.Conv2d(32, num_classes2, kernel_size=1, padding=0, stride=1, bias=True))


    def forward(self, x, x_coarse, x_fea):

        x=self.conv1_1(x)
        x_coarse=self.conv1_2(x_coarse)
        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)

        _, x_fine = self.net(x_fuse.unsqueeze(0), x_coarse.unsqueeze(0))
        x_fine = self.conv3(x_fine[0])

        return x_fine

class Trans_infer_rnn(nn.Module):
    def __init__(self, num_classes1, num_classes2, rnn_type='RNN'):
        super(Trans_infer_rnn, self).__init__()
        self.rnn_type=rnn_type
        if self.rnn_type=='rnn':
            self.net = convolutional_rnn.Conv2dRNN(in_channels=64 + num_classes1, out_channels=num_classes2,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=False,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type=='gru':
            self.net = convolutional_rnn.Conv2dGRU(in_channels=64 + num_classes1, out_channels=num_classes2,
                                                   kernel_size=1,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.,
                                                   bidirectional=False,
                                                   stride=1,
                                                   dilation=1,
                                                   groups=1)
        if self.rnn_type=='lstm':
            self.net = convolutional_rnn.Conv2dLSTM(in_channels=64 + num_classes1, out_channels=num_classes2,
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

    def forward(self, x, x_coarse, x_fea):

        x_fea = self.conv2(x_fea)
        x_fuse = torch.cat([x, x_fea], dim=1)

        _, x_fine = self.net(x_fuse.unsqueeze(0), x_coarse.unsqueeze(0))

        return x_fine[0]



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
            # self.PartHalfInfer = Trans_infer_rnn(part_cls, hbody_cls, self.trans_unit)
            # self.HalfFullInfer = Trans_infer_rnn(hbody_cls, fbody_cls, self.trans_unit)
            # self.FullHalfInfer = Trans_infer_rnn(fbody_cls, hbody_cls, self.trans_unit)
            # self.HalfPartInfer = Trans_infer_rnn(hbody_cls, part_cls, self.trans_unit)

            self.PartHalfInfer = Trans_infer_rnn_ed(part_cls, hbody_cls, self.trans_unit)
            self.HalfFullInfer = Trans_infer_rnn_ed(hbody_cls, fbody_cls, self.trans_unit)
            self.FullHalfInfer = Trans_infer_rnn_ed(fbody_cls, hbody_cls, self.trans_unit)
            self.HalfPartInfer = Trans_infer_rnn_ed(hbody_cls, part_cls, self.trans_unit)


    def forward(self, x_part, x_hbody, x_fbody, p_fea, h_fea, f_fea):
        x_hbody = self.PartHalfInfer(x_part, x_hbody, h_fea)
        x_fbody = self.HalfFullInfer(x_hbody, x_fbody, f_fea)
        x_hbody = self.FullHalfInfer(x_fbody, x_hbody, h_fea)
        x_part = self.HalfPartInfer(x_hbody, x_part, p_fea)

        return x_part, x_hbody, x_fbody


class Decoder(nn.Module):
    def __init__(self, num_classes=7, iter_step=1, trans_step=1, fbody_cls=2, hbody_cls=3):
        super(Decoder, self).__init__()
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.layer5 = SEOCModule(2048, 256, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = HBodyDecoder(hbody_cls=hbody_cls)
        self.layerf = FBodyDecoder(fbody_cls=fbody_cls)

        self.iter_step = iter_step
        self.trans_step = trans_step
        self.iter_trans = IterTrans(self.trans_step, 'rnn', fbody_cls, hbody_cls, part_cls=num_classes)


    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, xp = self.layer6(seg, x[1], x[0])
        x_hbody, xh = self.layerh(seg, x[0])
        x_fbody, xf = self.layerf(seg, x[0])
        for i in range(self.iter_step):
            x_seg, x_hbody, x_fbody = self.iter_trans(x_seg, x_hbody, x_fbody, xp, xh, xf)
        return [x_seg, x_hbody, x_fbody, x_dsn]


class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        self.encoder = ResNet(block, layers)
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
