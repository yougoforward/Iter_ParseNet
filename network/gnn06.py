import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
from modules.convGRU import ConvGRU
from modules.dcn import DFConv2d

class Composition(nn.Module):
    def __init__(self, hidden_dim):
        super(Composition, self).__init__()
        self.conv_ch = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.node_att = node_att()
    def forward(self, xh, xp_list):
        xp_att_list = [self.node_att(xp) for xp in xp_list]
        com_att = torch.max(torch.stack(xp_att_list, dim=1), dim=1, keepdim=False)[0]
        xph_message = sum([self.conv_ch(torch.cat([xh, xp*com_att], dim=1)) for xp in xp_list])
        return xph_message

class Decomposition(nn.Module):
    def __init__(self, hidden_dim=10, parts=2):
        super(Decomposition, self).__init__()
        self.conv_fh = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        self.decomp_att = Decomp_att(hidden_dim=hidden_dim, parts=parts)
        self.att = node_att()

    def forward(self, xf, xh_list):
        dec_att_list = self.decomp_att(xf,xh_list)
        xf_att = self.att(xf)
        decomp_fh_list = [self.conv_fh(torch.cat([xf*dec_att_list[i]*xf_att, xh_list[i]], dim=1)) for i in range(len(xh_list))]
        return decomp_fh_list

class Decomp_att(nn.Module):
    def __init__(self, hidden_dim=10, parts=2):
        super(Decomp_att, self).__init__()
        self.conv_fh = nn.Sequential(
            nn.Conv2d((parts+1) * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, parts, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, xf ,xh_list):
        decomp_att = self.conv_fh(torch.cat([xf]+xh_list, dim=1))
        decomp_att_list = list(torch.split(decomp_att, 1, dim=1))

        return decomp_att_list
class node_att(nn.Module):
    def __init__(self):
        super(node_att, self).__init__()

    def forward(self, xf):
        xff = xf*xf
        xff_sum = torch.sum(xff, dim=1, keepdim=True)
        parent_att = xff_sum/torch.max(xff_sum)
        return parent_att

class deformable_dense_Context(nn.Module):
    def __init__(self, hidden_dim=10):
        super(deformable_dense_Context, self).__init__()
        self.dcn_dilated1 = nn.Sequential(
                DFConv2d(
                    hidden_dim,
                    hidden_dim,
                    with_modulated_dcn=True,
                    kernel_size=3,
                    stride=1,
                    groups=1,
                    dilation=2,
                    deformable_groups=1,
                    bias=False
                ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.dcn_dilated2 = nn.Sequential(
            DFConv2d(
                hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=4,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.dcn_dilated3 = nn.Sequential(
            DFConv2d(
                hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=8,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

    def forward(self, hu):
        d_hu1 = self.dcn_dilated1(hu)
        d_hu_add1 = d_hu1+hu
        d_hu2 = self.dcn_dilated2(d_hu_add1)
        d_hu_add2 = d_hu2+d_hu_add1
        d_hu3 = self.dcn_dilated3(d_hu_add2)
        d_hu_add3 = d_hu3+d_hu_add2
        return d_hu_add3
# class deformable_dense_Context(nn.Module):
#     def __init__(self, hidden_dim=10):
#         super(deformable_dense_Context, self).__init__()
#         self.dcn_dilated1 = nn.Sequential(
#                 DFConv2d(
#                     hidden_dim,
#                     hidden_dim,
#                     with_modulated_dcn=True,
#                     kernel_size=3,
#                     stride=1,
#                     groups=1,
#                     dilation=1,
#                     deformable_groups=1,
#                     bias=False
#                 ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
#         self.dcn_dilated2 = nn.Sequential(
#             DFConv2d(
#                 hidden_dim,
#                 hidden_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
#     def forward(self, hu):
#         d_hu1 = self.dcn_dilated1(hu)
#         d_hu2 = self.dcn_dilated2(d_hu1)
#         return d_hu2
class Contexture(nn.Module):
    def __init__(self, hidden_dim=10, parts=6):
        super(Contexture, self).__init__()

        self.F_cont = nn.ModuleList(
            [deformable_dense_Context(hidden_dim) for i in range(parts)]
        )
        self.parts = parts
        self.A_att = node_att()

    def forward(self, xp_list):
        F_dep_list = [self.F_cont[i](xp_list[i].contiguous())*(1-self.A_att(xp_list[i])) for i in range(self.parts)]
        return F_dep_list

class Part_Dependency(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Part_Dependency, self).__init__()
        self.R_dep = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, F_dep_hu, hv):
        # F_dep = self.F_cont(hu)*(1-self.A_att(hu))
        huv = self.R_dep(torch.cat([F_dep_hu, hv], dim=1))
        return huv

class conv_Update(nn.Module):
    def __init__(self, hidden_dim=10):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        dtype = torch.cuda.FloatTensor
        self.update = ConvGRU(input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        kernel_size=(1,1),
                        num_layers=1,
                        dtype=dtype,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False)

    def forward(self, x, message):
        _, out = self.update(message.unsqueeze(1), [x])
        return out[0][0]

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
        self.comp_h = Composition(hidden_dim)
        self.conv_Update = conv_Update(hidden_dim)

    def forward(self, xf, xh_list, xp_list):
        comp_h = self.comp_h(xf, xh_list)
        xf =self.conv_Update(xf,comp_h)
        return xf


class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10, cls_p=7,
                 cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()
        self.cls_h = cls_h
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.hidden = hidden_dim

        self.decomp_fh_list = Decomposition(hidden_dim, parts=2)
        self.comp_u = Composition(hidden_dim)
        self.comp_l = Composition(hidden_dim)

        self.update_u = conv_Update(hidden_dim)
        self.update_l = conv_Update(hidden_dim)

    def forward(self, xf, xh_list, xp_list):
        decomp_list = self.decomp_fh_list(xf, xh_list)
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part - 1])

        comp_u = self.comp_u(xh_list[0], upper_parts)
        message_u = decomp_list[0]+comp_u
        xh_u = self.update_u(xh_list[0], message_u)

        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        comp_l = self.comp_l(xh_list[1], lower_parts)
        message_l = decomp_list[1]+comp_l
        xh_l = self.update_l(xh_list[1], message_l)

        xh_list_new = [xh_u, xh_l]
        return xh_list_new

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

        self.decomp_hpu_list = Decomposition(hidden_dim, parts=len(upper_part_list))
        self.decomp_hpl_list = Decomposition(hidden_dim, parts=len(lower_part_list))
        self.F_dep_list = Contexture(hidden_dim=hidden_dim, parts=self.cls_p - 1)
        self.part_dp_list = nn.ModuleList([Part_Dependency(hidden_dim) for i in range(self.edge_index_num)])
        self.node_update_list = nn.ModuleList([conv_Update(hidden_dim) for i in range(self.cls_p - 1)])

    def forward(self, xf, xh_list, xp_list):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part - 1])
        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])
        decomp_pu_list = self.decomp_hpu_list(xh_list[0], upper_parts )
        decomp_pl_list = self.decomp_hpl_list(xh_list[1], lower_parts )

        # F_dep_list = self.F_dep_list(xp_list)
        # xpp_list_list = [[] for i in range(self.cls_p - 1)]
        # for i in range(self.edge_index_num):
        #     xpp_list_list[self.edge_index[i, 1]].append(self.part_dp_list[self.edge_index[i, 1]](F_dep_list[self.edge_index[i, 0]],xp_list[self.edge_index[i, 1]]))

        xp_list_new = []
        for i in range(self.cls_p - 1):
            if i + 1 in self.upper_part_list:
                # message = decomp_pu_list[self.upper_part_list.index(i+1)]+sum(xpp_list_list[i])

                message = decomp_pu_list[self.upper_part_list.index(i + 1)]
            elif i + 1 in self.lower_part_list:
                # message = decomp_pl_list[self.lower_part_list.index(i+1)]+sum(xpp_list_list[i])

                message = decomp_pl_list[self.lower_part_list.index(i + 1)]
            xp_list_new.append(self.node_update_list[i](xp_list[i], message))
        return xp_list_new


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
        xh_list_new = self.half_infer(xf, xh_list, xp_list)
        # for part node
        xp_list_new = self.part_infer(xf, xh_list, xp_list)

        return xp_list_new, xh_list_new, xf_new


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
        self.node_cls_final = nn.Conv2d(hidden_dim*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))
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
        p_fea_list_new, h_fea_list_new, f_fea_new = self.gnn(p_node_list, h_node_list, f_node)
        p_fea_list_new2, h_fea_list_new2, f_fea_new2 = self.gnn(p_fea_list_new, h_fea_list_new, f_fea_new)

        # bg_node_new = self.bg_conv_new(torch.cat(p_fea_list_new + h_fea_list_new + [f_fea_new, bg_node], dim=1))

        # node supervision
        node = torch.cat([f_node] + h_node_list + p_node_list, dim=1)
        node_new = torch.cat([f_fea_new] + h_fea_list_new + p_fea_list_new, dim=1)
        node_new2 = torch.cat([f_fea_new2] + h_fea_list_new2 + p_fea_list_new2, dim=1)
        node_final = torch.cat([bg_node, node + node_new + node_new2], dim=1)
        node_seg_final = self.node_cls_final(node_final)
        node_seg_list = list(torch.split(node_seg_final, 1, dim=1))
        f_seg = torch.cat(node_seg_list[0:2], dim=1)
        h_seg = torch.cat([node_seg_list[0]] + node_seg_list[2:4], dim=1)
        p_seg = torch.cat([node_seg_list[0]] + node_seg_list[4:], dim=1)
        # p_seg_final = self.final_cls(p_seg, xp, xl)
        return p_seg, h_seg, f_seg


class Final_classifer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=20,  cls_p=7, cls_h=3, cls_f=2):
        super(Final_classifer, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim

        # classifier
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(cls_p+in_dim + 48, 256, kernel_size=3, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, cls_p, kernel_size=1, padding=0, dilation=1, bias=True),
                                   )
    def forward(self, p_seg, xp,  xl):
        # classifier
        _, _, th, tw = xl.size()
        xt = F.interpolate(torch.cat([p_seg, xp], dim=1), size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        p_seg_new = self.conv3(x)

        return p_seg_new

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
                                   in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2)
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
        p_seg, h_seg, f_seg = self.gnn_infer(x_fea, alpha_hb_fea, alpha_fb_fea, x[0])
        return p_seg, h_seg, f_seg, x_dsn


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
