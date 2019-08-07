import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
affine_par = True


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
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
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x2, x3, x4, x5]


class PSPModule(nn.Module):
    """Pyramid Pooling Module"""

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.psp1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(sizes[0], sizes[0])),
                                  nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                  InPlaceABNSync(out_features))

        self.psp2 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(sizes[1], sizes[1])),
                                  nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                  InPlaceABNSync(out_features))

        self.psp3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(sizes[2], sizes[2])),
                                  nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                  InPlaceABNSync(out_features))

        self.psp4 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(sizes[3], sizes[3])),
                                  nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                  InPlaceABNSync(out_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_features), nn.ReLU(inplace=False))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        prior1 = F.interpolate(self.psp1(feats), size=(h, w), mode='bilinear', align_corners=True)
        prior2 = F.interpolate(self.psp2(feats), size=(h, w), mode='bilinear', align_corners=True)
        prior3 = F.interpolate(self.psp3(feats), size=(h, w), mode='bilinear', align_corners=True)
        prior4 = F.interpolate(self.psp4(feats), size=(h, w), mode='bilinear', align_corners=True)
        bottle = self.bottleneck(torch.cat([prior1, prior2, prior3, prior4, feats], dim=1))
        return bottle


class Decoder(nn.Module):
    """Decoder Network"""

    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.layer5 = PSPModule(2048, 512)
        self.layer6 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm2d(256), nn.ReLU(inplace=False),
                                    nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        x_seg = self.layer5(x[-1])
        x_seg = self.layer6(x_seg)
        return [x_seg, x_dsn]


class Network(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(Network, self).__init__()
        self.encoder = ResNet(block, layers)
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
    model = Network(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
