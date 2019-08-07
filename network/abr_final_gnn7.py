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
from modules.gnn_infer import Decomposition, Composition, conv_Update

class Part_Dependency(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Dependency, self).__init__()
        self.cls_p = cls_p

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

        self.B_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, p_fea, pA, pB, A_att, B_att):
        A_diffuse = self.dconv(torch.cat([pB, pA], dim=1))
        # A_att = self.A_att(pA)
        A_diffuse_att = (1 - A_att) * A_diffuse
        # B_att = self.B_att(pB)
        A2B = A_diffuse_att*B_att
        return A2B

class DecoderModule(nn.Module):
    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        xt_fea = self.conv1(xt)
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

        self.comp =Composition(hidden_dim, cls_h-1)
        self.conv_Update = conv_Update(hidden_dim, 1)

    def forward(self, xf, xh_list):
        message=self.comp(xh_list, xf)
        xf = self.conv_Update(xf, [message])

        return xf, message


class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()

        self.upper_part_list= upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.hidden = hidden_dim
        self.decomp_fhu = Decomposition(in_dim, hidden_dim)
        self.decomp_fhl = Decomposition(in_dim, hidden_dim)

        self.part_dp_h = Part_Dependency(in_dim, hidden_dim)
        self.part_dp_l = Part_Dependency(in_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Conv2d((cls_h - 1) * hidden_dim, cls_h - 1, kernel_size=1, padding=0, stride=1, bias=True,
                      groups=cls_h - 1),
            nn.Sigmoid())
        self.comp_phu = Composition(hidden_dim, self.upper_parts_len)
        self.comp_phl = Composition(hidden_dim, self.lower_parts_len)

        self.update_u = conv_Update(hidden_dim, 3)
        self.update_l = conv_Update(hidden_dim, 3)

    def forward(self,h_fea, xh_list, xf, xp_list):
        dp_att_list=torch.split(self.att(torch.cat(xh_list, dim=1)), 1, dim=1)

        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part-1])
        xphu = self.comp_phu(upper_parts, xh_list[0])
        xlh = self.part_dp_h(h_fea, xh_list[1], xh_list[0], dp_att_list[1], dp_att_list[0])
        xfhu, att_u = self.decomp_fhu(h_fea, xf, xh_list[0])

        message_u = [xphu, xlh, xfhu]
        xh_u = self.update_u(xh_list[0], message_u)

        #lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        xphl = self.comp_phl(lower_parts, xh_list[1])
        xuh = self.part_dp_l(h_fea,xh_list[0], xh_list[1], dp_att_list[0], dp_att_list[1])
        xfhl, att_l = self.decomp_fhl(h_fea, xf, xh_list[1])

        message_l = [xphl, xuh, xfhl]
        xh_l = self.update_l(xh_list[1], message_l)

        xh_list_new = [xh_u,xh_l]
        att_list =[att_u, att_l]
        return xh_list_new, att_list, dp_att_list

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]


        self.part_dp_list = nn.ModuleList([Part_Dependency(in_dim, hidden_dim) for i in range(self.edge_index_num)])
        self.att = nn.Sequential(
            nn.Conv2d((cls_p - 1) * hidden_dim, cls_p - 1, kernel_size=1, padding=0, stride=1, bias=True,
                      groups=cls_p - 1),
            nn.Sigmoid())

        self.decomp_hp_list = nn.ModuleList([Decomposition(in_dim, hidden_dim) for i in range(cls_p-1)])
        self.update_conv_list = nn.ModuleList(
            [conv_Update(hidden_dim, 2) for i in range(cls_p - 1)])

    def forward(self, p_fea, xp_list, xh_list):
        xhp_list = []
        att_list = []
        for i in range(1, self.cls_p):
            if i in self.upper_part_list:
                xhp, att = self.decomp_hp_list[i - 1](p_fea, xh_list[0], xp_list[i - 1])
                xhp_list.append(xhp)
                att_list.append(att)
            elif i in self.lower_part_list:
                xhp, att = self.decomp_hp_list[i - 1](p_fea, xh_list[1], xp_list[i - 1])
                xhp_list.append(xhp)
                att_list.append(att)

        dp_att_list = torch.split(self.att(torch.cat(xp_list, dim=1)), 1, dim=1)
        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        xpp_list = []
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(
                self.part_dp_list[i](p_fea, xp_list[self.edge_index[i, 0]], xp_list[self.edge_index[i, 1]], dp_att_list[self.edge_index[i, 0]], dp_att_list[self.edge_index[i, 1]]))

        for i in range(self.cls_p - 1):
            if len(xpp_list_list[i]) == 1:
                xpp_list.append(xpp_list_list[i][0])
            else:
                xpp_list.append(torch.max(torch.stack(xpp_list_list[i], dim=-1), dim=-1, keepdim=False)[0])

        message_list = [[xpp_list[j], xhp_list[j]] for j in range(self.cls_p - 1)]

        xp_list = [self.update_conv_list[j](xp_list[j], message_list[j]) for j in range(self.cls_p - 1)]

        return xp_list, att_list, dp_att_list


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

        # for full body node
        xf_new, xf_message = self.full_infer(xf, xh_list)
        # for half body node
        xh_list_new, fh_att_list, h_att_list = self.half_infer(h_fea, xh_list, xf, xp_list)
        # for part node
        xp_list_new, hp_att_list, p_att_list = self.part_infer(p_fea, xp_list, xh_list)

        att = (torch.cat(hp_att_list+fh_att_list, dim=1)+torch.cat(p_att_list+h_att_list, dim=1))/2.0
        return xp_list_new, xh_list_new, xf_new, att


class GNN_infer(nn.Module):
    def __init__(self, adj_matrix, upper_half_node =[1,2,3,4], lower_half_node = [5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # feature transform
        self.p_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_p-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_p-1)), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_h-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_h-1)), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim * (cls_f-1), kernel_size=1, padding=0, stride=1, bias=False),
                                    BatchNorm2d(hidden_dim * (cls_f-1)), nn.ReLU(inplace=False))
        self.bg_conv = nn.Sequential(
            nn.Conv2d((cls_p+cls_h+cls_f-3)*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.bg_conv_new = nn.Sequential(
            nn.Conv2d((cls_p+cls_h+cls_f-3)*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn=GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p, self.cls_h, self.cls_f)

        #node supervision
        # multi-label classifier
        self.node_cls = nn.Conv2d(hidden_dim * (cls_p + cls_h + cls_f - 2), (cls_p + cls_h + cls_f - 2), kernel_size=1,
                                  padding=0, stride=1, bias=True, groups=(cls_p + cls_h + cls_f - 2))
        self.node_cls_new = nn.Conv2d(hidden_dim * (cls_p + cls_h + cls_f - 2), (cls_p + cls_h + cls_f - 2),
                                      kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p + cls_h + cls_f - 2))
        # self.node_cls_new2 = nn.Conv2d(self.hidden*(cls_p+cls_h+cls_f-2), (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=(cls_p+cls_h+cls_f-2))

        # self.final_cls = nn.Sequential(nn.Conv2d((cls_p+cls_h+cls_f-2)*2*hidden_dim, (cls_p+cls_h+cls_f-2)*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False, groups=(cls_p+cls_h+cls_f-2)),
        #                                BatchNorm2d((cls_p+cls_h+cls_f-2)*hidden_dim), nn.ReLU(inplace=False),
        #                                nn.Conv2d((cls_p+cls_h+cls_f-2)*hidden_dim, (cls_p+cls_h+cls_f-2), kernel_size=1, padding=0, stride=1, bias=True, groups=1))

        self.p_cls = nn.Conv2d(hidden_dim * (cls_p * 2), cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=1)
        self.h_cls = nn.Conv2d(hidden_dim * (cls_h * 2), cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=1)
        self.f_cls = nn.Conv2d(hidden_dim * (cls_f * 2), cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=1)
    def forward(self, xp, xh, xf):
        # feature transform
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))
        f_node = self.f_conv(xf)
        bg_node = self.bg_conv(torch.cat(p_node_list+h_node_list+[f_node], dim=1))

        # gnn infer
        p_fea_list_new, h_fea_list_new, f_fea_new, att = self.gnn(p_node_list, h_node_list, f_node, bg_node, xp, xh, xf)
        bg_node_new = self.bg_conv_new(torch.cat(p_fea_list_new+h_fea_list_new+[f_fea_new], dim=1))

        #node supervision
        node = torch.cat([bg_node, f_node] + h_node_list + p_node_list, dim=1)
        node_seg = self.node_cls(node)
        node_new = torch.cat([bg_node_new, f_fea_new] + h_fea_list_new + p_fea_list_new, dim=1)
        node_seg_final = self.node_cls_new(node_new)
        # node_seg_new2 = self.node_cls_new2(torch.cat([bg_node_new2, f_fea_new2] + h_fea_list_new2 + p_fea_list_new2, dim=1))

        node_seg = sum([node_seg, node_seg_final]) / 2.0
        node_att = att

        node_seg_list = list(torch.split(node_seg, 1, dim=1))
        f_seg = torch.cat(node_seg_list[0:2], dim=1)
        h_seg = torch.cat([node_seg_list[0]]+node_seg_list[2:4], dim=1)
        p_seg = torch.cat([node_seg_list[0]]+node_seg_list[4:], dim=1)

        batch, c, h, w = node.size()
        xphf_infer = torch.stack([node, node_new], dim=2).view(batch, 2*c, h, w)
        xphf_infer_list = list(torch.split(xphf_infer, self.hidden_dim*2, dim=1))
        xf_infer = torch.cat(xphf_infer_list[0:2], dim=1)
        xh_infer = torch.cat([xphf_infer_list[0]] + xphf_infer_list[2:4], dim=1)
        xp_infer = torch.cat([xphf_infer_list[0]] + xphf_infer_list[4:], dim=1)
        p_seg_final = self.p_cls(xp_infer)
        h_seg_final = self.h_cls(xh_infer)
        f_seg_final = self.f_cls(xf_infer)

        # final_seg = self.final_cls(xphf_infer)
        # final_seg_list = list(torch.split(final_seg, 1, dim=1))
        #
        # f_seg_final = torch.cat(final_seg_list[0:2], dim=1)
        # h_seg_final = torch.cat([final_seg_list[0]] + final_seg_list[2:4], dim=1)
        # p_seg_final = torch.cat([final_seg_list[0]] + final_seg_list[4:], dim=1)

        # xphf_infer =torch.cat([node, node_new], dim=1)
        # return xphf_infer, p_seg, h_seg, f_seg, node_seg, node_att

        return p_seg_final, h_seg_final, f_seg_final, p_seg, h_seg, f_seg, node_seg, node_att

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

        self.h_cls = nn.Sequential(nn.Conv2d(in_dim, cls_h, kernel_size=1, padding=0, stride=1, bias=True))

        self.f_cls = nn.Sequential(nn.Conv2d(in_dim, cls_f, kernel_size=1, padding=0, stride=1, bias=True))

    def forward(self, xp, xh, xf, xl):
        # classifier
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


class fuse_DecoderModule(nn.Module):

    def __init__(self, hidden=20, num_classes=7, cls_h=3, cls_f=2):
        super(fuse_DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(2*hidden * (num_classes+cls_h+cls_f-2), 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)
                                   )

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)
                                   )

    def forward(self, x1, x2, x3, xl):
        xt = self.conv0(torch.cat([x1, x2, x3], dim=1))
        _, _, th, tw = xl.size()
        xt = F.interpolate(xt, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        x_seg = self.conv4(x_fea)
        h_seg = self.conv5(x_fea)
        f_seg = self.conv6(x_fea)
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
        self.gnn_infer=GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1,2,3,4], lower_half_node=[5,6], in_dim=256, hidden_dim=20, cls_p=7, cls_h=3, cls_f=2)
        #
        # self.classifier = Final_classifer(in_dim=256, cls_p=7, cls_h=3, cls_f=2)
        self.fuse_seg = fuse_DecoderModule(hidden=20, num_classes=7, cls_h=3, cls_f=2)

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
        p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg, node_seg, node_att =self.gnn_infer(x_fea,alpha_hb_fea,alpha_fb_fea)
        # p_seg, h_seg, f_seg = self.classifier(xp_seg, xh_seg, xf_seg, x[0])
        # p_seg, h_seg, f_seg = self.fuse_seg(xphf_seg, x[0])
        return p_seg, h_seg, f_seg, pg_seg, hg_seg, fg_seg, node_seg, node_att, x_dsn


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
