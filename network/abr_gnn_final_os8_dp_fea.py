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
from modules.gnn_infer import Decomposition, Composition

class Part_Dependency(nn.Module):
    def __init__(self,hidden_dim=10):
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
            ),BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

        self.A_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, pA, pB):
        A_diffuse = self.dconv(torch.cat([pA, pB], dim=1))
        A_att = self.A_att(pA)
        A_diffuse_att = (2 - A_att) * A_diffuse
        return A_diffuse_att

class conv_Update(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, paths_len=3):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_update1 = nn.Sequential(
            DFConv2d(
                (paths_len+1) * hidden_dim+in_dim,
                2 * hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False)
        )
        self.conv_update2 = nn.Sequential(
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

        self.gate = nn.Sequential(
            DFConv2d(
                2 * hidden_dim,
                1,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()

    def forward(self, fea, x, message_list):
        if len(message_list)>1:
            out = self.conv_update1(torch.cat([fea, x]+message_list, dim=1))
            gate = self.gate(out)
            out = self.conv_update2(out)
        else:
            out = self.conv_update1(torch.cat([fea, x, message_list[0]], dim=1))
            gate = self.gate(out)
            out = self.conv_update2(out)
        return self.relu(gate*x+(1-gate)*out)

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
        # return x_fea


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

        self.comp_h = Composition(hidden_dim, cls_h - 1)
        self.comp_p = Composition(hidden_dim, cls_p - 1)
        self.conv_Update = conv_Update(in_dim, hidden_dim, 2)

    def forward(self, fea, xf, xh_list, xp_list):
        _, _, h, w = xf.size()
        comp_h = self.comp_h(xh_list)
        comp_p = self.comp_p(xp_list)
        xf = self.conv_Update(fea, xf, [comp_h, comp_p])
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
        self.decomp_fhu = Decomposition(hidden_dim)
        self.comp_phu = Composition(hidden_dim, self.upper_parts_len)

        self.part_dp_u = Part_Dependency(hidden_dim)
        self.part_dp_l = Part_Dependency(hidden_dim)


        self.decomp_fhl = Decomposition(hidden_dim)
        self.comp_phl = Composition(hidden_dim, self.lower_parts_len)

        self.update_u = conv_Update(in_dim, hidden_dim, 3)
        self.update_l = conv_Update(in_dim, hidden_dim, 3)

    def forward(self, fea, xf, xh_list, xp_list):
        _, _, h, w = xf.size()
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part - 1])
        comp_u = self.comp_phu(upper_parts)
        dp_u =self.part_dp_u(xh_list[1], xh_list[0])

        att_u = self.decomp_fhu(xf, xh_list[0])
        decomp_u = att_u*xf
        message_u = [comp_u, decomp_u, dp_u]

        xh_u = self.update_u(fea, xh_list[0], message_u)

        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        comp_l = self.comp_phl(lower_parts)
        dp_l = self.part_dp_l(xh_list[0], xh_list[1])
        att_l = self.decomp_fhl(xf, xh_list[1])
        decomp_l = att_l*xf
        message_l = [comp_l, decomp_l, dp_l]

        xh_l = self.update_u(fea, xh_list[1], message_l)

        xh_list_new = [xh_u, xh_l]
        att_list = [att_u, att_l]
        att_h = torch.cat(att_list, dim=1)
        return xh_list_new, att_h

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]

        self.part_dp_list = nn.ModuleList([Part_Dependency(hidden_dim) for i in range(self.edge_index_num)])

        self.decomp_hp_list = nn.ModuleList([Decomposition(hidden_dim) for i in range(cls_p - 1)])
        self.decomp_fp_list = nn.ModuleList([Decomposition(hidden_dim) for i in range(cls_p - 1)])
        self.update_conv_list = nn.ModuleList(
            [conv_Update(in_dim, hidden_dim, 3) for i in range(cls_p - 1)])

    def forward(self, fea, xf, xh_list, xp_list):
        decomp_fp = []
        decomp_hp = []
        att_fp = []
        att_hp = []
        # _, _, h, w = xp_list[0].size()
        # xf = F.interpolate(xf, (h,w))
        # xh_0 = F.interpolate(xh_list[0], (h,w), mode ="bilinear", align_corners=True)
        # xh_1 = F.interpolate(xh_list[1], (h,w), mode ="bilinear", align_corners=True)
        # xh_list = [xh_0, xh_1]

        for i in range(self.cls_p-1):
            att_f = self.decomp_fp_list[i](xf, xp_list[i])
            decomp_fp.append(att_f * xf)
            if i+1 in self.upper_part_list:
                att_h = self.decomp_hp_list[i](xh_list[0], xp_list[i])
                decomp_hp.append(att_h*xh_list[0])
            elif i+1 in self.lower_part_list:
                att_h = self.decomp_hp_list[i](xh_list[1], xp_list[i])
                decomp_hp.append(att_h*xh_list[1])

            att_fp.append(att_f)
            att_hp.append(att_h)

        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        xpp_list = []
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(
                self.part_dp_list[i](xp_list[self.edge_index[i, 0]], xp_list[self.edge_index[i, 1]]))
        for i in range(self.cls_p - 1):
            if len(xpp_list_list[i]) == 1:
                xpp_list.append(xpp_list_list[i][0])
            else:
                xpp_list.append(sum(xpp_list_list[i]))

        message_list = [[decomp_fp[j], decomp_hp[j], xpp_list[j]] for j in range(self.cls_p - 1)]

        xp_list = [self.update_conv_list[j](fea, xp_list[j], message_list[j]) for j in range(self.cls_p - 1)]
        att_p = (torch.cat(att_fp, dim=1)+torch.cat(att_hp, dim=1))/2.0
        return xp_list, att_p

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

    def forward(self, xp_fea, xh_fea, xf_fea, xp_list, xh_list, xf):
        _, _, h, w = xp_list[0].size()
        # for full body node
        xf_new = self.full_infer(xf_fea, xf, xh_list, xp_list)
        # for half body node
        xh_list_new, att_h = self.half_infer(xh_fea, xf, xh_list, xp_list)
        # for part node
        xp_list_new, att_p = self.part_infer(xp_fea, xf, xh_list, xp_list)

        att = torch.cat([att_h, att_p], dim=1)
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
        self.p_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_p), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_p)), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_h), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_h)), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_f), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_f)), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn=GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p, self.cls_h, self.cls_f)

        # node supervision
        # node supervision
        # classifier
        self.pg_cls = nn.Conv2d(self.hidden_dim * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.hg_cls = nn.Conv2d(self.hidden_dim * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.fg_cls = nn.Conv2d(self.hidden_dim * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)

        self.pg_cls_new = nn.Conv2d(self.hidden_dim * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_p)
        self.hg_cls_new = nn.Conv2d(self.hidden_dim * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_h)
        self.fg_cls_new = nn.Conv2d(self.hidden_dim * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True,
                                    groups=cls_f)

        # multi-label classifier
        self.node_cls = nn.Conv2d(hidden_dim*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))
        self.node_cls_new = nn.Conv2d(hidden_dim*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))
        # self.node_cls_new2 = nn.Conv2d(self.hidden*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))

        self.final_cls = nn.Sequential(nn.Conv2d((cls_p+cls_h+cls_f-2)*2*hidden_dim, (cls_p+cls_h+cls_f-2)*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False, groups=(cls_p+cls_h+cls_f-2)),
                                       BatchNorm2d((cls_p+cls_h+cls_f-2)*hidden_dim), nn.ReLU(inplace=False),
                                       nn.Conv2d((cls_p+cls_h+cls_f-2)*hidden_dim, (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2)))


    def forward(self, xp, xh, xf):
        _, _, th, tw = xp.size()
        # feature transform
        p_node_pred = self.p_conv(xp)
        h_node_pred = self.h_conv(xh)
        f_node_pred = self.f_conv(xf)

        p_nodes = list(torch.split(p_node_pred, self.hidden_dim, dim=1))
        h_nodes = list(torch.split(h_node_pred, self.hidden_dim, dim=1))
        f_nodes = list(torch.split(f_node_pred, self.hidden_dim, dim=1))
        p_node_list = p_nodes[1:]
        h_node_list = h_nodes[1:]
        f_node = f_nodes[1]

        # gnn infer
        p_fea_list_new, h_fea_list_new, f_fea_new, att = self.gnn(xp, xh, xf, p_node_list, h_node_list, f_node)

        # node supervision
        pg_seg = self.pg_cls(p_node_pred)
        hg_seg = self.hg_cls(h_node_pred)
        fg_seg = self.fg_cls(f_node_pred)

        xp_seg = torch.cat([p_nodes[0]] + p_fea_list_new, dim=1)
        xh_seg = torch.cat([h_nodes[0]] + h_fea_list_new, dim=1)
        xf_seg = torch.cat([f_nodes[0]] + [f_fea_new], dim=1)
        pg_seg_new = self.pg_cls(xp_seg)
        hg_seg_new = self.hg_cls(xh_seg)
        fg_seg_new = self.fg_cls(xf_seg)

        xphf_infer =torch.cat([xp_seg, xh_seg, xf_seg], dim=1)
        p_seg = sum([pg_seg, pg_seg_new])
        h_seg = sum([hg_seg, hg_seg_new])
        f_seg = sum([fg_seg, fg_seg_new])
        return xphf_infer, p_seg, h_seg, f_seg, att

class fuse_DecoderModule(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=20, cls_p=7, cls_h=3, cls_f=2):
        super(fuse_DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(DFConv2d(
                in_dim+(cls_p + cls_h + cls_f) * hidden_dim,
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
            ), BatchNorm2d(in_dim), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(in_dim+48, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False)
                                   )
        self.p_cls = nn.Sequential(
                                   nn.Conv2d(in_dim, cls_p, kernel_size=1, padding=0, dilation=1, bias=True)
                                   )

        # self.p_cls = nn.Sequential(nn.Conv2d(in_dim * 3 + (cls_p + cls_h + cls_f - 2) * hidden_dim, cls_p, kernel_size=1, padding=0, stride=1, bias=True))
        self.h_cls = nn.Sequential(
            nn.Conv2d(in_dim + (cls_p + cls_h + cls_f ) * hidden_dim, cls_h, kernel_size=1, padding=0, stride=1,
                      bias=True))
        self.f_cls = nn.Sequential(
            nn.Conv2d(in_dim + (cls_p + cls_h + cls_f) * hidden_dim, cls_f, kernel_size=1, padding=0, stride=1,
                      bias=True))

    def forward(self, xphf, xp, xh, xf, xl):
        _, _, th, tw = xl.size()
        xt = self.conv0(F.interpolate(torch.cat([xphf, xp], dim=1), size=(th, tw), mode='bilinear', align_corners=True))
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)

        x_seg = self.p_cls(x_fea)
        h_seg = self.h_cls(torch.cat([xphf, xh], dim=1))
        f_seg = self.f_cls(torch.cat([xphf, xf], dim=1))
        return x_seg, h_seg, f_seg

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)
        #
        self.adj_matrix=torch.tensor([[0,1,0,0,0,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,1],[0,0,0,0,1,0]], requires_grad=False)
        self.gnn_infer=GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1,2,3,4], lower_half_node=[5,6], in_dim=256, hidden_dim=40, cls_p=num_classes, cls_h=hbody_cls, cls_f=fbody_cls)
        #
        self.fuse_seg = fuse_DecoderModule(in_dim=256, hidden_dim=40, cls_p=num_classes, cls_h=hbody_cls, cls_f=fbody_cls)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        # direct infer
        x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb_fea= self.layerh(seg, x[1])
        alpha_fb_fea = self.layerf(seg, x[1])

        # gnn infer
        xphf_seg, pg_seg, hg_seg, fg_seg, node_att =self.gnn_infer(x_fea,alpha_hb_fea,alpha_fb_fea)
        p_seg, h_seg, f_seg = self.fuse_seg(xphf_seg, x_fea, alpha_hb_fea, alpha_fb_fea, x[0])
        return p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg, node_att, x_dsn


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
