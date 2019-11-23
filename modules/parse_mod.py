import functools

import torch
from torch import nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import SEModule, ContextContrastedModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class SA_upsample(nn.Module):
    """The basic implementation for self-attention block/non-local block
    Parameters:
        in_dim       : the dimension of the input feature map
        key_dim      : the dimension after the key/query transform
        value_dim    : the dimension after the value transform
        scale        : choose the scale to downsample the input feature maps (save memory cost)
    """

    def __init__(self, in_dim, out_dim, key_dim, value_dim):
        super(SA_upsample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.func_key = nn.Sequential(nn.Conv2d(in_channels=self.in_dim, out_channels=self.key_dim,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      InPlaceABNSync(self.key_dim))
        self.func_query = self.func_key
        self.func_value = nn.Conv2d(in_channels=self.in_dim, out_channels=self.value_dim,
                                    kernel_size=1, stride=1, padding=0)

        self.refine = nn.Sequential(nn.Conv2d(self.value_dim, self.out_dim, kernel_size=1, padding=0, bias=False),
                                    InPlaceABNSync(out_dim))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, xl):
        _, _, hl, wl = xl.size()
        xl = F.interpolate(x, size=(hl, wl), mode='bilinear', align_corners=True) + self.alpha*xl
        batch, h, w = x.size(0), x.size(2), x.size(3)

        value = self.func_value(x).view(batch, self.value_dim, -1)  # bottom
        value = value.permute(0, 2, 1)
        query = self.func_query(xl).view(batch, self.key_dim, -1)  # top
        query = query.permute(0, 2, 1)
        key = self.func_key(x).view(batch, self.key_dim, -1)  # mid

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch, self.value_dim, *xl.size()[2:])
        output = self.refine(xl + self.beta*context)
        return output

class SelfAttentionModule(nn.Module):
    """The basic implementation for self-attention block/non-local block
    Parameters:
        in_dim       : the dimension of the input feature map
        key_dim      : the dimension after the key/query transform
        value_dim    : the dimension after the value transform
        scale        : choose the scale to downsample the input feature maps (save memory cost)
    """

    def __init__(self, in_dim, out_dim, key_dim, value_dim, scale=2):
        super(SelfAttentionModule, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.func_key = nn.Sequential(nn.Conv2d(in_channels=self.in_dim, out_channels=self.key_dim,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      InPlaceABNSync(self.key_dim))
        self.func_query = self.func_key
        self.func_value = nn.Conv2d(in_channels=self.in_dim, out_channels=self.value_dim,
                                    kernel_size=1, stride=1, padding=0)
        self.weights = nn.Conv2d(in_channels=self.value_dim, out_channels=self.out_dim,
                                 kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.weights.weight, 0)
        nn.init.constant_(self.weights.bias, 0)

        self.refine = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                    InPlaceABNSync(out_dim))

    def forward(self, x):
        batch, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.func_value(x).view(batch, self.value_dim, -1)  # bottom
        value = value.permute(0, 2, 1)
        query = self.func_query(x).view(batch, self.key_dim, -1)  # top
        query = query.permute(0, 2, 1)
        key = self.func_key(x).view(batch, self.key_dim, -1)  # mid

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch, self.value_dim, *x.size()[2:])
        context = self.weights(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        output = self.refine(context)
        return output


class ChannelAttentionModule(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(ChannelAttentionModule, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, chn, height, width = x.size()
        proj_query = x.view(m_batchsize, chn, -1)
        proj_key = x.view(m_batchsize, chn, -1).permute(0, 2, 1)
        energy = torch.matmul(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, chn, -1)

        out = torch.matmul(attention, proj_value)
        out = out.view(m_batchsize, chn, height, width)

        out = self.gamma * out + x
        return out

class ASPPModule(nn.Module):
    """ASPP with OC module: aspp + oc context"""

    def __init__(self, in_dim, out_dim):
        super(ASPPModule, self).__init__()

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_dim, out_dim, 1, bias=False), InPlaceABNSync(out_dim))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=6, dilation=6, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=18, dilation=18, bias=False),
                                        InPlaceABNSync(out_dim))

        # self.dilation_4 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
        #                                 InPlaceABNSync(out_dim),
        #                                 nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=24, dilation=24, bias=False),
        #                                 InPlaceABNSync(out_dim))

        # self.psaa_conv = nn.Sequential(nn.Conv2d(in_dim + 6 * out_dim, out_dim, 1, padding=0, bias=False),
        #                                 InPlaceABNSync(out_dim),
        #                                 nn.Conv2d(out_dim, 6, 1, bias=True))

        # self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 6, out_dim, kernel_size=1, padding=0, bias=False),
        #                                InPlaceABNSync(out_dim))
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_dim + 5 * out_dim, out_dim, 1, padding=0, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, 5, 1, bias=True))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        _,_,h,w = x.size()
        feat0 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        # feat5 = self.dilation_4(x)
        # fusion branch
        # concat = torch.cat([feat0, feat1, feat2, feat3, feat4], 1)
        # output = self.head_conv(concat)

        # psaa
        # y1 = torch.cat((feat0, feat1, feat2, feat3, feat4, feat5), 1)
        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        psaa_feat = self.psaa_conv(torch.cat([x, y1], dim=1))
        psaa_att = torch.sigmoid(psaa_feat)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        # y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
        #                 psaa_att_list[3] * feat3, psaa_att_list[4] * feat4, psaa_att_list[5]*feat5), 1)
        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
                        psaa_att_list[3] * feat3, psaa_att_list[4] * feat4), 1)
        output = self.head_conv(y2)
        return output


class ASPAtteModule(nn.Module):
    """ASPP with OC module: aspp + oc context"""

    def __init__(self, in_dim, out_dim, scale):
        super(ASPAtteModule, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                                         InPlaceABNSync(out_dim),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, scale=scale))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=24, dilation=24, bias=False),
                                        InPlaceABNSync(out_dim))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=36, dilation=36, bias=False),
                                        InPlaceABNSync(out_dim))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim),
                                       nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        # fusion branch
        concat = torch.cat([feat0, feat1, feat2, feat3, feat4], 1)
        output = self.head_conv(concat)
        return output


class SEASPPModule(nn.Module):
    """ASPP based on SE and OC
    best [1, 12, 24, 36]
    current [1, 6, 12, 18]
    """

    def __init__(self, in_dim, out_dim, scale):
        super(SEASPPModule, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=True),
                                         InPlaceABNSync(out_dim),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, scale=scale))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=24, dilation=24, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=36, dilation=36, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim),
                                       nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        # fusion branch
        concat = torch.cat([feat0, feat1, feat2, feat3, feat4], 1)
        output = self.head_conv(concat)
        return output


class MagicModule(nn.Module):
    """ASPP based on SE and OC and Context Contrasted """

    def __init__(self, in_dim, out_dim, scale):
        super(MagicModule, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                                         InPlaceABNSync(out_dim),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, scale=scale))
        # TODO: change SE Module to Channel Attention Module
        self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        # self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
        #                                 InPlaceABNSync(out_dim), ChannelAttentionModule(out_dim))

        self.dilation_0 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=6),
                                        SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=12),
                                        SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=18),
                                        SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=24),
                                        SEModule(out_dim, reduction=16))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 6, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim),
                                       nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        featx = self.dilation_x(x)
        # fusion branch
        concat = torch.cat([feat0, feat1, feat2, feat3, feat4, featx], 1)
        output = self.head_conv(concat)
        return output


class SE_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(SE_Module, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(in_dim, in_dim // 16, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.ReLU(False),
                                nn.Conv2d(in_dim // 16, out_dim, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        out = self.se(x)
        return out


class sa_MagicModule(nn.Module):
    """ASPP based on SE and OC and Context Contrasted """

    def __init__(self, in_dim, out_dim, scale):
        super(sa_MagicModule, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                                         InPlaceABNSync(out_dim),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, scale=scale))
        # TODO: change SE Module to Channel Attention Module
        self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        # self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
        #                                 InPlaceABNSync(out_dim), ChannelAttentionModule(out_dim))

        self.dilation_0 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=6),
                                        SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=12),
                                        SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=18),
                                        SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=24),
                                        SEModule(out_dim, reduction=16))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 6, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim)
                                       )
        self.refine = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(out_dim))
        self.project = nn.Conv2d(6 * out_dim, 6, kernel_size=1, padding=0, bias=True)
                               

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        featx = self.dilation_x(x)

        n, c, h, w = feat0.size()

        # fusion branch
        concat = torch.cat([feat0, feat1, feat2, feat3, feat4, featx], 1)
        output = self.head_conv(concat)

        # scale adaptie
        energy = self.project(concat)
        attention = self.softmax(energy)
        y = torch.stack([feat0, feat1, feat2, feat3, feat4, featx], dim=-1)
        out = torch.matmul(y.view(n, c, h*w, 6).permute(0,2,1,3), attention.view(n, 5, h*w).permute(0,2,1).unsqueeze(dim=3))
        out = out.squeeze(dim=3).permute(0,2,1).view(n,c,h,w)

        # refine 
        output = self.refine(out+output)
        return output