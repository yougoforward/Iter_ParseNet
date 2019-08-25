import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

from modules.dcn import DFConv2d

class Decomposition(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(Decomposition, self).__init__()
        self.att_fh = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.LeakyReLU(inplace=False))
        self.att_fh1=nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, xf, xh):
        att_fh = self.att_fh(torch.cat([xf, xh], dim=1))
        att = self.att_fh1(att_fh)
        return att

class conv_Update(nn.Module):
    def __init__(self, hidden_dim=10, paths_len=3):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_update = nn.Sequential(
            DFConv2d(
                (paths_len+1) * hidden_dim,
                2 * hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            DFConv2d(
                2 * hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()

    def forward(self, x, message_list):
        if len(message_list)>1:
            out = self.conv_update(torch.cat([x]+message_list, dim=1))
        else:
            out = self.conv_update(torch.cat([x, message_list[0]], dim=1))
        return self.relu(self.gamma*x+out)

class Part_Dependency(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Part_Dependency, self).__init__()
        self.dconv = nn.Sequential(
            DFConv2d(
                2 * hidden_dim,
                2 * hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            DFConv2d(
                2 * hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

        self.A_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, pA, pB):
        A_diffuse = self.dconv(torch.cat([pA, pB], dim=1))
        A_att = self.A_att(pA)
        A_diffuse_att = (2 - A_att) * A_diffuse
        return A_diffuse_att


class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        # self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(48), nn.ReLU(inplace=False))
        #
        # self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False),
        #                            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
        #                            BatchNorm2d(256), nn.ReLU(inplace=False))

        # self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        _, _, th, tw = xl.size()
        xt_fea = self.conv1(xt)
        # xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        # xl = self.conv2(xl)
        # x = torch.cat([xt, xl], dim=1)
        # x_fea = self.conv3(x)
        # x_seg = self.conv4(x_fea)
        return xt_fea


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        xfuse = self.conv1(xfuse)

        return xfuse


class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))
        self.alpha_fb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_fb * skip
        xfuse = self.conv1(xfuse)
        return xfuse

class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.hidden = hidden_dim
        self.conv_Update = conv_Update(hidden_dim, cls_h+cls_p-2)

    def forward(self, xf, xh_list, xp_list):
        xf = self.conv_Update(xf, xh_list+xp_list)
        return xf


class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10, cls_p=7,
                 cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()

        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.hidden = hidden_dim

        self.decomp_u = Decomposition(in_dim, hidden_dim)
        self.decomp_l = Decomposition(in_dim, hidden_dim)

        self.update_u = conv_Update(hidden_dim, 2+self.upper_parts_len)
        self.update_l = conv_Update(hidden_dim, 2+self.lower_parts_len)

    def forward(self, xf, xh_list, xp_list):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part - 1])

        att_fhu = self.decomp_u(xf, xh_list[0])
        message_u = [xf]+upper_parts+[xh_list[1]]
        xh_u = self.update_u(xh_list[0], message_u)

        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        att_fhl = self.decomp_l(xf, xh_list[0])
        message_l = [xf]+lower_parts+[xh_list[0]]
        xh_l = self.update_l(xh_list[1], message_l)

        att_fh_list = [att_fhu, att_fhl]
        xh_list_new = [xh_u, xh_l]
        return xh_list_new, att_fh_list

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                    cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.xpp_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            self.xpp_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.decomp_fp_list = nn.ModuleList([Decomposition(in_dim, hidden_dim) for i in range(cls_p - 1)])
        self.decomp_hp_list = nn.ModuleList([Decomposition(in_dim, hidden_dim) for i in range(cls_p - 1)])

        self.update_conv_list = nn.ModuleList(
            [conv_Update(hidden_dim, 2+len(self.xpp_list_list[i])) for i in range(cls_p - 1)])

    def forward(self, xf, xh_list, xp_list):
        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(xp_list[self.edge_index[i, 0]])

        att_fp_list = []
        att_hp_list = []
        xp_list_new = []
        for i in range(self.cls_p-1):
            if i+1 in self.upper_part_list:
                att_fp = self.decomp_fp_list[i](xf, xp_list[i])
                att_hp = self.decomp_hp_list[i](xh_list[0], xp_list[i])
                xp_list_new.append(self.update_conv_list[i](xp_list[i], [xf, xh_list[0]]+xpp_list_list[i]))
            elif i+1 in self.lower_part_list:
                att_fp = self.decomp_fp_list[i](xf, xp_list[i])
                att_hp = self.decomp_hp_list[i](xh_list[1], xp_list[i])
                xp_list_new.append(self.update_conv_list[i](xp_list[i], [xf, xh_list[1]]+xpp_list_list[i]))
            att_fp_list.append(att_fp)
            att_hp_list.append(att_hp)
        return xp_list_new, att_fp_list, att_hp_list

class GNN(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim
        self.upper_half_node = upper_half_node
        self.upper_node_len = len(self.upper_half_node)
        self.lower_half_node = lower_half_node
        self.lower_node_len = len(self.lower_half_node)

        self.full_infer = Full_Graph(in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.half_infer = Half_Graph(self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h,
                                     cls_f)
        self.part_infer = Part_Graph(adj_matrix, self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p,
                                     cls_h, cls_f)

    def forward(self, xp_list, xh_list, xf):
        # for full body node
        xf_new = self.full_infer(xf, xh_list, xp_list)
        # for half body node
        xh_list_new, att_fh_list = self.half_infer(xf, xh_list, xp_list)
        # for part node
        xp_list_new, att_fp_list, att_hp_list = self.part_infer(xf, xh_list, xp_list)

        att = torch.cat([torch.cat(att_fh_list, dim=1), (torch.cat(att_fp_list, dim=1)+torch.cat(att_hp_list, dim=1))/2.0], dim=1)
        # att = (torch.cat(hp_att_list+fh_att_list, dim=1)+torch.cat(p_att_list+h_att_list, dim=1))/2.0

        return xp_list_new, xh_list_new, xf_new, att


class GNN_infer(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # feature transform
        self.p_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * (cls_p - 1), kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * (cls_p - 1)), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * (cls_h - 1), kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * (cls_h - 1)), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * (cls_f - 1), kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * (cls_f - 1)), nn.ReLU(inplace=False))
        self.bg_conv = nn.Sequential(
            nn.Conv2d(3 * in_dim, hidden_dim, kernel_size=1, padding=0, stride=1,
                      bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        # self.bg_conv_new = nn.Sequential(
        #     nn.Conv2d((cls_p + cls_h + cls_f - 2) * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1,
        #               bias=False),
        #     BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn = GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p,
                       self.cls_h, self.cls_f)

        # node supervision
        # multi-label classifier
        self.node_cls = nn.Conv2d(hidden_dim*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))
        self.node_cls_new = nn.Conv2d(hidden_dim*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))
        # self.node_cls_new2 = nn.Conv2d(self.hidden*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))

        self.final_cls = Final_classifer(in_dim, hidden_dim, cls_p, cls_h, cls_f)


    def forward(self, xp, xh, xf, xl):
        # _, _, th, tw = xp.size()
        # _, _, h, w = xh.size()
        #
        # xh = F.interpolate(xh, (th, tw), mode='bilinear', align_corners=True)
        # xf = F.interpolate(xf, (th, tw), mode='bilinear', align_corners=True)
        # feature transform
        f_node = self.f_conv(xf)
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))
        bg_node = self.bg_conv(torch.cat([xp, xh, xf], dim=1))



        # gnn infer
        p_fea_list_new, h_fea_list_new, f_fea_new, att_decomp = self.gnn(p_node_list, h_node_list, f_node)
        # bg_node_new = self.bg_conv_new(torch.cat(p_fea_list_new + h_fea_list_new + [f_fea_new, bg_node], dim=1))

        # node supervision
        node = torch.cat([bg_node, f_node] + h_node_list + p_node_list, dim=1)
        node_seg = self.node_cls(node)
        node_new = torch.cat([bg_node, f_fea_new] + h_fea_list_new + p_fea_list_new, dim=1)
        node_seg_final = self.node_cls_new(node_new)
        # node_seg_new2 = self.node_cls_new2(torch.cat([bg_node_new2, f_fea_new2] + h_fea_list_new2 + p_fea_list_new2, dim=1))

        node_seg = sum([node_seg, node_seg_final]) / 2.0

        node_seg_list = list(torch.split(node_seg, 1, dim=1))
        f_seg = torch.cat(node_seg_list[0:2], dim=1)
        h_seg = torch.cat([node_seg_list[0]] + node_seg_list[2:4], dim=1)
        p_seg = torch.cat([node_seg_list[0]] + node_seg_list[4:], dim=1)

        # xphf_infer =torch.cat([node, node_new], dim=1)
        xphf_infer = node_new
        p_seg_final, h_seg_final, f_seg_final = self.final_cls(xphf_infer, xp, xh, xf, xl)

        return p_seg_final, h_seg_final, f_seg_final, p_seg, h_seg, f_seg, att_decomp

# class Final_classifer(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=20,  cls_p=7, cls_h=3, cls_f=2):
#         super(Final_classifer, self).__init__()
#         self.cp = cls_p
#         self.ch = cls_h
#         self.cf = cls_f
#         self.ch_in = in_dim

#         # classifier
#         self.p_cls = nn.Sequential(DFConv2d(
#                 in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim,
#                 in_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(in_dim), nn.ReLU(inplace=False),
#             DFConv2d(
#                 in_dim,
#                 in_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(in_dim), nn.ReLU(inplace=False),
#             nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True))
#         # self.p_cls = nn.Sequential(nn.Conv2d(in_dim * 3 + (cls_p + cls_h + cls_f - 2) * hidden_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True))
#         self.h_cls = nn.Sequential(nn.Conv2d(in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True))
#         self.f_cls = nn.Sequential(nn.Conv2d(in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True))

#     def forward(self, xphf, xp, xh, xf):
#         # classifier
#         xp_seg = self.p_cls(torch.cat([xphf, xp, xh, xf], dim=1))
#         xh_seg = self.h_cls(torch.cat([xphf, xp, xh, xf], dim=1))
#         xf_seg = self.f_cls(torch.cat([xphf, xp, xh, xf], dim=1))

#         return xp_seg, xh_seg, xf_seg
class Final_classifer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=20,  cls_p=7, cls_h=3, cls_f=2):
        super(Final_classifer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim

        # classifier
        self.conv0 = nn.Sequential(DFConv2d(
                in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim,
                in_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(in_dim), nn.ReLU(inplace=False),
            DFConv2d(
                in_dim,
                in_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(in_dim), nn.ReLU(inplace=False)
        )

        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(in_dim + 48, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim)
                                   )
        self.relu = nn.ReLU(inplace=False)
        self.p_cls = nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)

        # self.p_cls = nn.Sequential(nn.Conv2d(in_dim * 3 + (cls_p + cls_h + cls_f - 2) * hidden_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True))
        self.h_cls = nn.Sequential(nn.Conv2d(in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True))
        self.f_cls = nn.Sequential(nn.Conv2d(in_dim*3+(cls_p + cls_h + cls_f - 2) * hidden_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, xphf, xp, xh, xf, xl):
        # classifier
        _, _, th, tw = xl.size()
        xt = F.interpolate(self.conv0(torch.cat([xphf, xp, xh, xf], dim=1)), size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.relu(self.conv3(x)+xt)

        xp_seg = self.p_cls(x_fea)
        xh_seg = self.h_cls(torch.cat([xphf, xp, xh, xf], dim=1))
        xf_seg = self.f_cls(torch.cat([xphf, xp, xh, xf], dim=1))

        return xp_seg, xh_seg, xf_seg
class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)
        #
        self.adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]], requires_grad=False)
        self.gnn_infer = GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6],
                                   in_dim=256, hidden_dim=20, cls_p=7, cls_h=3, cls_f=2)
        #
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        # direct infer
        x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb_fea = self.layerh(seg, x[1])
        alpha_fb_fea = self.layerf(seg, x[1])

        # gnn infer
        p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg, att = self.gnn_infer(x_fea, alpha_hb_fea, alpha_fb_fea, x[0])
        return p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg, att, x_dsn


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
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)  # 101
    # model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
