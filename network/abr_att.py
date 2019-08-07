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
class Part_Dependency(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Part_Dependency, self).__init__()
        self.dconv = nn.Sequential(
            DFConv2d(
            2*hidden_dim,
            2*hidden_dim,
            with_modulated_dcn=True,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            deformable_groups=1,
            bias=False
            ),BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            DFConv2d(
                2* hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ),BatchNorm2d(hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.A_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())
        # self.message = nn.Sequential(
        #     nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=True),
        #     BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

    def forward(self, pA, pB):
        AB_att = self.dconv(torch.cat([pA, pB], dim=1))
        A_att = self.A_att(pA)
        # A2B = self.message((AB_att*p_fea+p_fea)*(1-A_att))
        # A2B = self.message(AB_att * p_fea * (1 - A_att))
        A2B = AB_att*(1-A_att)
        return A2B


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

        xt_fea = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)

        x = torch.cat([xt_fea, xl], dim=1)
        x_fea = self.conv3(x)
        x_seg = self.conv4(x_fea)
        xt_fea = F.interpolate(x_fea, size=(h, w), mode='bilinear', align_corners=True)

        return x_seg, xt_fea

class fuse_DecoderModule(nn.Module):

    def __init__(self, indim=256, hidden=10, num_classes=7, cls_h=3, cls_f=2):
        super(fuse_DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(indim*3+(num_classes+cls_h+cls_f-2)*hidden, indim, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(indim), nn.ReLU(inplace=False),
                                   nn.Conv2d(indim, indim, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(indim), nn.ReLU(inplace=False))

        self.att_p = nn.Sequential(
            nn.Conv2d(indim+num_classes*hidden, 1, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.Sigmoid())
        self.att_h = nn.Sequential(
            nn.Conv2d(indim + cls_h*hidden, 1, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.Sigmoid())
        self.att_f = nn.Sequential(
            nn.Conv2d(indim + cls_f*hidden, 1, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.Sigmoid())
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)

        self.conv6 = nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, p_fea_list, h_fea_list, f_fea, bg_node, xp, xh, xf):
        total_fea = torch.cat(p_fea_list+h_fea_list+[f_fea, bg_node, xp, xh, xf], dim=1)
        xt_fea = self.conv0(total_fea)

        att_p = self.att_p(torch.cat(p_fea_list+[bg_node, xp], dim=1))
        att_h = self.att_h(torch.cat(h_fea_list+[bg_node, xh], dim=1))
        att_f = self.att_f(torch.cat([f_fea, bg_node, xf], dim=1))

        p_seg = self.conv4(xt_fea)
        h_seg = self.conv5(xt_fea)
        f_seg = self.conv6(xt_fea)
        return p_seg*att_p, h_seg*att_h, f_seg*att_f


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

        self.conv2 =nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)

        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        xfuse = self.conv1(xfuse)

        output = self.conv2(xfuse)
        return output, xfuse

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
        xfuse = self.conv1(xfuse)
        output = self.conv2(xfuse)
        return output, xfuse


class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.hidden = hidden_dim

        self.comp = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.self_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.att_update = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, f_fea, xf, xh_list):
        comp_att = self.comp(torch.cat(xh_list, dim=1))
        self_att = self.self_att(xf)
        xf = self.att_update(f_fea*(1+comp_att+self_att))

        return xf


class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()

        self.upper_part_list= upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.hidden = hidden_dim

        self.part_dp_u = Part_Dependency(hidden_dim)
        self.part_dp_l = Part_Dependency(hidden_dim)

        self.decomp_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.comp_att_u = nn.Sequential(
            nn.Conv2d(self.upper_parts_len * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.self_att_u = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.att_update_u = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

        self.comp_att_l = nn.Sequential(
            nn.Conv2d(self.lower_parts_len * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.self_att_l = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.att_update_l = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )


    def forward(self,h_fea, xh_list, xf, xp_list):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part-1])

        decomp_att = self.decomp_att(xf)

        comp_att_u = self.comp_att_u(torch.cat(upper_parts, dim=1))
        xlh_att = self.part_dp_u(xh_list[1], xh_list[0])
        self_att_u = self.self_att_u(xh_list[0])

        xh_u = self.att_update_u(h_fea*(1+comp_att_u+decomp_att+self_att_u+xlh_att))

        #lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        comp_att_l = self.comp_att_l(torch.cat(lower_parts, dim=1))
        xuh_att = self.part_dp_l(xh_list[0], xh_list[1])
        self_att_l = self.self_att_l(xh_list[1])

        xh_l = self.att_update_l(h_fea * (1 + comp_att_l + decomp_att + self_att_l + xuh_att))

        xh_list_new = [xh_u,xh_l]
        return xh_list_new

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]


        self.decomp_att_u = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.decomp_att_l = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.att_update_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(cls_p - 1)])

        self.part_dp_list = nn.ModuleList([Part_Dependency(hidden_dim) for i in range(self.edge_index_num)])
        self.att = nn.Sequential(
            nn.Conv2d((cls_p - 1) * hidden_dim, cls_p - 1, kernel_size=1, padding=0, stride=1, bias=True,
                      groups=cls_p - 1),
            nn.Sigmoid())


    def forward(self, p_fea, xp_list, xh_list):
        self_att_list=torch.split(self.att(torch.cat(xp_list, dim=1)), 1, dim=1)
        xpp_list_list = [[] for i in range(self.cls_p-1)]
        xpp_list=[]
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i,1]].append(self.part_dp_list[i](xp_list[self.edge_index[i,0]], xp_list[self.edge_index[i,1]]))

        for i in range(self.cls_p-1):
            if len(xpp_list_list[i])==1:
                xpp_list.append(xpp_list_list[i][0])
            else:
                xpp_list.append(sum(xpp_list_list[i]))


        decomp_att_u = self.decomp_att_u(xh_list[0])
        decomp_att_l = self.decomp_att_l(xh_list[1])
        update_list = []
        for i in range(1, self.cls_p):
            if i in self.upper_part_list:
                update_list.append(self.att_update_list[i - 1](p_fea*(1+decomp_att_u+self_att_list[i-1]+xpp_list[i-1])))
            elif i in self.lower_part_list:
                update_list.append(self.att_update_list[i - 1](p_fea*(1+decomp_att_l+self_att_list[i-1]+xpp_list[i-1])))

        return update_list

class GNN(nn.Module):
    def __init__(self, adj_matrix, upper_half_node =[1,2,3,4], lower_half_node = [5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
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
        self.half_infer = Half_Graph(self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.part_infer = Part_Graph(adj_matrix,self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h, cls_f)


    def forward(self, xp_list, xh_list, xf, bg_node, p_fea, h_fea, f_fea):
        xf_new = self.full_infer(f_fea, xf, xh_list)
        xh_list_new = self.half_infer(h_fea, xh_list, xf, xp_list)
        xp_list_new = self.part_infer(p_fea, xp_list, xh_list)

        return xp_list_new, xh_list_new, xf_new

# class fuse_DecoderModule(nn.Module):
#
#     def __init__(self, num_classes=7, cls_h=3, cls_f=2):
#         super(fuse_DecoderModule, self).__init__()
#         self.conv0 = nn.Sequential(nn.Conv2d(256*3, 256, kernel_size=3, padding=1, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))
#         self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))
#
#         self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(48), nn.ReLU(inplace=False))
#
#         self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False),
#                                    nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))
#
#         self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
#         self.conv5 = nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)
#
#         self.conv6 = nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)
#
#     def forward(self, x1, x2, x3, xl):
#         xt = self.conv0(torch.cat([x1,x2,x3], dim=1))
#         xt_fea = self.conv1(xt)
#
#         _, _, th, tw = xl.size()
#         xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
#         xl = self.conv2(xl)
#         x = torch.cat([xt, xl], dim=1)
#         x_fea = self.conv3(x)
#         x_seg = self.conv4(x_fea+xt)
#         h_seg = self.conv5(xt_fea)
#         f_seg = self.conv6(xt_fea)
#         return x_seg, h_seg, f_seg

class GNN_infer(nn.Module):
    def __init__(self, adj_matrix, upper_half_node =[1,2,3,4], lower_half_node = [5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
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
        self.gnn=GNN(adj_matrix, upper_half_node, lower_half_node, self.ch_in, self.hidden, self.cp, self.ch, self.cf)

        # node supervision
        # classifier
        self.pg_cls = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.hg_cls = nn.Conv2d(self.hidden * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.fg_cls = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)

        self.pg_cls_new = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_p)
        self.hg_cls_new = nn.Conv2d(self.hidden * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_h)
        self.fg_cls_new = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_f)

        self.relu = nn.ReLU(inplace=False)

        self.fuse_seg = fuse_DecoderModule(in_dim, self.hidden, num_classes=7, cls_h=3, cls_f=2)


    def forward(self, xp, xh, xf):
        # feature transform
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden, dim=1))
        f_node = self.f_conv(xf)
        bg_node = self.bg_conv(torch.cat([xp, xh, xf], dim=1))

        # gnn infer
        p_fea_list_new, h_fea_list_new, f_fea_new = self.gnn(p_node_list, h_node_list, f_node, bg_node, xp, xh, xf)

        # node supervision
        pg_seg = self.pg_cls(torch.cat([bg_node] + p_node_list, dim=1))
        hg_seg = self.hg_cls(torch.cat([bg_node] + h_node_list, dim=1))
        fg_seg = self.fg_cls(torch.cat([bg_node, f_node], dim=1))

        pg_seg_new = self.pg_cls_new(torch.cat([bg_node] + p_fea_list_new, dim=1))
        hg_seg_new = self.hg_cls_new(torch.cat([bg_node] + h_fea_list_new, dim=1))
        fg_seg_new = self.fg_cls_new(torch.cat([bg_node] + [f_fea_new], dim=1))

        p_seg, h_seg, f_seg = self.fuse_seg(p_fea_list_new, h_fea_list_new, f_fea_new, bg_node, xp, xh, xf)

        return p_seg, h_seg, f_seg, pg_seg_new+pg_seg, hg_seg_new+hg_seg, fg_seg_new+fg_seg


class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)

        #
        self.adj_matrix=torch.tensor([[0,1,0,0,0,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,1],[0,0,0,0,1,0]], requires_grad=False)
        self.gnn_infer=GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1,2,3,4], lower_half_node=[5,6], in_dim=256, hidden_dim=20, cls_p=7, cls_h=3, cls_f=2)
        #
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, h_fea= self.layerh(seg, x[1])
        alpha_fb, f_fea = self.layerf(seg, x[1])

        p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg = self.gnn_infer(x_fea, h_fea, f_fea)
        _, _, h, w = x[0].size()
        x_part = x_seg + F.interpolate(p_seg, size=(h, w), mode='bilinear', align_corners=True)
        x_hb = alpha_hb+h_seg
        x_fb = alpha_fb+f_seg

        return [x_part, x_hb, x_fb, pg_seg, hg_seg, fg_seg, x_dsn]



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
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes) #101
    # model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
