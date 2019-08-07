import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154


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
        # self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False),
        #                            nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))


        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   SEModule(256, reduction=16))
        self.conv2 =nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)

        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        # _, _, h, w = skip.size()
        #
        # xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # xfuse = xup + self.alpha_hb * skip
        xfuse = self.conv1(x)

        output = self.conv2(xfuse)
        return output, xfuse

class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False),
        #                            nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   SEModule(256, reduction=16))
        self.conv2 = nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        # _, _, h, w = skip.size()
        #
        # xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # xfuse = xup + self.alpha_fb * skip
        xfuse = self.conv1(x)
        output = self.conv2(xfuse)
        return output, xfuse


class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.hidden = hidden_dim

        self.conv_hf = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_update = nn.Sequential(
            nn.Conv2d(2*hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
            )

    def forward(self, xf, xh):
        message=self.conv_hf(torch.cat(xh, dim=1))
        xf = self.conv_update(torch.cat([xf, message], dim=1))

        return xf, message


class Half_upper_Graph(nn.Module):
    def __init__(self, part_list=[1,2,3,4], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Half_upper_Graph, self).__init__()

        self.part_list= [int(i) for i in part_list]
        self.parts = len(part_list)
        self.hidden = hidden_dim


        self.conv_ph = nn.Sequential(
            nn.Conv2d(self.parts * hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(in_dim + hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_update = nn.Sequential(
            nn.Conv2d(2*hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self,h_fea, xh_list, xfh, xp_list):
        parts = []
        for part in self.part_list:
            parts.append(xp_list[part-1])
        xph = self.conv_ph(torch.cat(parts, dim=1))
        xlh = self.conv_lh(torch.cat([h_fea, xh_list[1]], dim=1))
        message = xph + xlh + xfh
        xh_u = self.conv_update(torch.cat([xh_list[0], message], dim=1))
        return xh_u, message


class Half_lower_Graph(nn.Module):
    def __init__(self, part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Half_lower_Graph, self).__init__()

        self.part_list = [int(i) for i in part_list]
        self.parts = len(part_list)

        self.conv_ph = nn.Sequential(
            nn.Conv2d(self.parts * hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(in_dim + hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_update = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, h_fea, xh_list, xfh, xp_list):
        parts = []
        for part in self.part_list:
            parts.append(xp_list[part-1])

        xph = self.conv_ph(torch.cat(parts, dim=1))
        xuh = self.conv_lh(torch.cat([h_fea, xh_list[0]], dim=1))
        message = xph+xuh+xfh
        xh_l = self.conv_update(torch.cat([xh_list[1], message], dim=1))

        return xh_l, message

class Part_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p

        self.node_conv_list = nn.ModuleList(
            [nn.Sequential(
            nn.Conv2d(in_dim + hidden_dim*(cls_p-2), 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(cls_p-1)])

        self.update_conv_list = nn.ModuleList(
            [nn.Sequential(
            nn.Conv2d(2*hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(cls_p-1)])

    def forward(self, p_fea, xp_list, xhp_list):

        xpp_list = [self.node_conv_list[j](torch.cat([p_fea]+[xp_list[i] for i in range(self.cls_p-1) if i!=j], dim=1)) for j in range(self.cls_p-1)]
        message_list = [xpp_list[j]+xhp_list[j] for j in range(self.cls_p - 1)]
        xp_list = [
            self.update_conv_list[j](torch.cat([xp_list[j], message_list[j]], dim=1))
            for j in range(self.cls_p - 1)]
        return xp_list, message_list


class GNN(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(GNN, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim
        self.upper_half_node = [1,2,3,4]
        self.upper_node_len = len(self.upper_half_node)
        self.lower_half_node = [5,6]
        self.lower_node_len = len(self.lower_half_node)

        self.full_infer = Full_Graph(in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.half_upper_infer = Half_upper_Graph(self.upper_half_node, in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.half_lower_infer = Half_lower_Graph(self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.part_infer = Part_Graph(in_dim, hidden_dim, cls_p, cls_h, cls_f)


        self.full_att =  nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.upper_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.lower_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.full_decomp = nn.Sequential(
            nn.Conv2d(in_dim, (cls_h-1) * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d((cls_h-1) * hidden_dim), nn.ReLU(inplace=False)
        )
        self.upper_half_decomp = nn.Sequential(
            nn.Conv2d(in_dim, self.upper_node_len * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(self.upper_node_len * hidden_dim), nn.ReLU(inplace=False)
        )
        self.lower_half_decomp = nn.Sequential(
            nn.Conv2d(in_dim, self.lower_node_len * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(self.lower_node_len * hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, xp_list, xh_list, xf, bg_node, p_fea, h_fea, f_fea):
        # for full body node
        xf_new, xf_message = self.full_infer(xf, xh_list)

        # for half body node
        xfh_list = list(torch.split(self.full_decomp(h_fea*self.full_att(xf)), self.hidden, dim=1))
        xhu_new, xhu_message = self.half_upper_infer(h_fea, xh_list, xfh_list[0], xp_list)
        xhl_new, xhl_message = self.half_lower_infer(h_fea, xh_list, xfh_list[1], xp_list)
        xh_list_new = [xhu_new, xhl_new]

        xh_message_list=[xhu_message, xhl_message]

        # for part node

        #upper half nodes
        upper_xhp_list = list(torch.split(self.upper_half_decomp(p_fea*self.upper_att(xh_list[0])), self.hidden,dim=1))
        # lower half nodes
        lower_xhp_list = list(
            torch.split(self.lower_half_decomp(p_fea*self.lower_att(xh_list[1])), self.hidden, dim=1))

        xhp_list=[]
        for i in range(1,self.cp):
            if i in self.upper_half_node:
                xhp_list.append(upper_xhp_list[self.upper_half_node.index(i)])
            elif i in self.lower_half_node:
                xhp_list.append(lower_xhp_list[self.lower_half_node.index(i)])

        xp_list_new, xp_message_list = self.part_infer(p_fea, xp_list, xhp_list)

        return xp_list_new, xh_list_new, xf_new, xf_message, xh_message_list, xp_message_list


class GNN_infer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim

        # feature transform
        self.p_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_p-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_p-1)), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_h-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_h-1)), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_f-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_f-1)), nn.ReLU(inplace=False))
        self.bg_conv = nn.Sequential(
            nn.Conv2d(3*in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
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

        # classifier
        # self.p_cls = nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True)
        # self.h_cls = nn.Conv2d(in_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True)
        # self.f_cls = nn.Conv2d(in_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True)

        #node supervision
        # classifier
        self.pg_cls = nn.Conv2d(self.hidden*cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.hg_cls = nn.Conv2d(self.hidden*cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.fg_cls = nn.Conv2d(self.hidden*cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)

        self.f_message_cls = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)
        self.h_message_cls = nn.Conv2d(self.hidden*cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.p_message_cls = nn.Conv2d(self.hidden*cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)

        self.pg_cls_new = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.hg_cls_new = nn.Conv2d(self.hidden * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.fg_cls_new = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)

        self.alpha_p = nn.Parameter(torch.ones(1))
        self.alpha_h = nn.Parameter(torch.ones(1))
        self.alpha_f = nn.Parameter(torch.ones(1))


    def forward(self, xp, xh, xf):
        # feature transform
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden, dim=1))
        f_node = self.f_conv(xf)
        bg_node = self.bg_conv(torch.cat([xp, xh, xf], dim=1))
        # gnn infer
        # p_h_fea_list, h_h_fea_list, f_h_fea_list = self.gnn(p_h_fea_list, h_h_fea_list, f_h_fea_list)
        p_fea_list_new, h_fea_list_new, f_fea_new, f_message, h_message_list, p_message_list = self.gnn(p_node_list, h_node_list, f_node, bg_node, xp, xh, xf)


        #node supervision
        pg_seg = self.pg_cls(torch.cat([bg_node]+p_node_list, dim=1))
        hg_seg = self.hg_cls(torch.cat([bg_node]+h_node_list, dim=1))
        fg_seg = self.fg_cls(torch.cat([bg_node, f_node], dim=1))

        f_message_seg = self.f_message_cls(torch.cat([bg_node, f_message], dim=1))
        h_message_seg = self.h_message_cls(torch.cat([bg_node]+h_message_list, dim=1))
        p_message_seg = self.p_message_cls(torch.cat([bg_node]+p_message_list, dim=1))

        pg_seg_new = self.pg_cls(torch.cat([bg_node]+p_fea_list_new, dim=1))
        hg_seg_new = self.hg_cls(torch.cat([bg_node]+h_fea_list_new, dim=1))
        fg_seg_new = self.fg_cls(torch.cat([bg_node]+[f_fea_new], dim=1))

        #feature d_transform
        #feature d_transform
        #feature d_transform

        # xp_infer = xp+self.alpha_p*self.p_dconv(torch.cat([bg_node]+p_fea_list_new, dim=1))
        # xh_infer = xh+self.alpha_h*self.h_dconv(torch.cat([bg_node]+h_fea_list_new, dim=1))
        # xf_infer = xf+self.alpha_f*self.f_dconv(torch.cat([bg_node, f_fea_new], dim=1))

        # xp_infer = torch.cat([bg_node]+p_fea_list_new, dim=1)
        # xh_infer = torch.cat([bg_node]+h_fea_list_new, dim=1)
        # xf_infer = torch.cat([bg_node, f_fea_new], dim=1)
        #classifier
        # xp_seg = self.p_cls(xp_infer)
        # xh_seg = self.h_cls(xh_infer)
        # xf_seg = self.f_cls(xf_infer)

        return pg_seg_new, hg_seg_new, fg_seg_new, pg_seg, hg_seg, fg_seg, p_message_seg, h_message_seg, f_message_seg


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
        outputs = []
        for ind, x in enumerate(inputs):
            _, _, ht, wt = inputs[ind].size()
            if ht!=h or wt!=w:
                up_x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
                outputs.append(up_x)
            else:
                outputs.append(x)

        # output = torch.cat(outputs, dim=1)
        # output = self.conv1(output)

        output = torch.stack(outputs, dim=1)
        output = torch.mean(output, dim=1, keepdim=False)
        return output

class Final_classifer(nn.Module):
    def __init__(self, in_dim=256, cls_p=7, cls_h=3, cls_f=2):
        super(Final_classifer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        # classifier
        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim + 48, in_dim, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(in_dim), nn.ReLU(inplace=False))

        self.p_cls = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True))

        self.h_cls = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True))

        self.f_cls = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True))


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

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)

        self.fuse_p = CombineBlock(num_classes, 3)
        self.fuse_h = CombineBlock(hbody_cls, 3)
        self.fuse_f = CombineBlock(fbody_cls, 3)


        self.gnn_infer=GNN_infer(in_dim=256, hidden_dim=20, cls_p=7, cls_h=3, cls_f=2)

        self.classifier = Final_classifer(in_dim=256, cls_p=7, cls_h=3, cls_f=2)
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, alpha_hb_fea= self.layerh(x_fea, x[1])
        alpha_fb, alpha_fb_fea = self.layerf(alpha_hb_fea, x[1])

        xp_seg, xh_seg, xf_seg, node_p_seg, node_h_seg, node_f_seg, node_p_message_seg, node_h_message_seg, node_f_message_seg =self.gnn_infer(x_fea,alpha_hb_fea,alpha_fb_fea)

        # p_seg, h_seg, f_seg=self.classifier(xp_seg, xh_seg, xf_seg, x[0])

        p_seg = self.fuse_p([x_seg, xp_seg, node_p_seg])
        h_seg = self.fuse_h([alpha_hb, xh_seg, node_h_seg])
        f_seg = self.fuse_f([alpha_fb, xf_seg, node_f_seg])
        # p_seg = x_seg+p_seg
        # h_seg = alpha_hb+h_seg
        # f_seg = alpha_fb+f_seg

        # return [p_seg, h_seg, f_seg, x_seg, alpha_hb, alpha_fb, node_p_seg, node_h_seg, node_f_seg, x_dsn]

        return [p_seg, h_seg, f_seg, node_p_message_seg, node_h_message_seg, node_f_message_seg, x_dsn]


class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        self.encoder = ResGridNet(block, layers)
        # self.encoder = senet154()
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
