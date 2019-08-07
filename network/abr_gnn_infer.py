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
from modules.gnn_infer import Composition, Decomposition, att_Update, conv_Update, Part_Dependency, Pair_Part_Dependency, conv_Update

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
        x_fea = F.interpolate(x_fea, size=(h, w), mode='bilinear', align_corners=True)
        return x_seg, x_fea


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

        self.comp =Composition(hidden_dim, cls_h-1)
        self.conv_Update = conv_Update(hidden_dim)

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

        self.comp_phu = Composition(hidden_dim, self.upper_parts_len)
        self.comp_phl = Composition(hidden_dim, self.lower_parts_len)

        self.part_dp = Pair_Part_Dependency()
        self.p_dp = Part_Dependency(in_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Conv2d((cls_h-1)*hidden_dim, cls_h-1, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h-1),
            nn.Sigmoid())

        self.update_u = conv_Update(hidden_dim)
        self.update_l = conv_Update(hidden_dim)

    def forward(self,h_fea, xh_list, xf, xp_list):
        part_dp_list = self.p_dp(h_fea, xh_list)
        att_list=torch.split(self.att(torch.cat(xh_list, dim=1)), 1, dim=1)
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(xp_list[part-1])

        xphu = self.comp_phu(upper_parts, xh_list[0])
        xlh = self.part_dp(part_dp_list[1], xh_list[0], att_list[1], att_list[0])
        xfhu,att_u = self.decomp_fhu(h_fea, xf, xh_list[0])

        message_u = [xphu, xlh, xfhu]
        xh_u = self.update_u(xh_list[0], message_u)

        #lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(xp_list[part - 1])

        xphl = self.comp_phl(lower_parts, xh_list[1])
        xuh = self.part_dp(part_dp_list[0], xh_list[1], att_list[0], att_list[1])
        xfhl, att_l = self.decomp_fhl(h_fea, xf, xh_list[1])

        message_l = [xphl, xuh, xfhl]
        xh_l = self.update_l(xh_list[1], message_l)

        xh_list_new = [xh_u,xh_l]
        att_list =[att_u, att_l]
        return xh_list_new, att_list

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]

        self.p_dp = Part_Dependency(in_dim, hidden_dim)
        self.part_dp = Pair_Part_Dependency()
        self.att = nn.Sequential(
            nn.Conv2d((cls_p - 1) * hidden_dim, cls_p - 1, kernel_size=1, padding=0, stride=1, bias=True,
                      groups=cls_p - 1),
            nn.Sigmoid())

        self.decomp_hp_list = nn.ModuleList([Decomposition(in_dim, hidden_dim) for i in range(cls_p-1)])
        self.update_conv_list = nn.ModuleList(
            [conv_Update(hidden_dim) for i in range(cls_p - 1)])

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


        part_dp_list = self.p_dp(p_fea, xp_list)
        self_att_list = torch.split(self.att(torch.cat(xp_list, dim=1)), 1, dim=1)
        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        xpp_list = []
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(
                self.part_dp(part_dp_list[self.edge_index[i, 0]], xp_list[self.edge_index[i, 1]],
                             self_att_list[self.edge_index[i, 0]], self_att_list[self.edge_index[i, 1]]))

        for i in range(self.cls_p - 1):
            if len(xpp_list_list[i]) == 1:
                xpp_list.append(xpp_list_list[i][0])
            else:
                xpp_list.append(torch.max(torch.stack(xpp_list_list[i], dim=-1), dim=-1, keepdim=False)[0])

        message_list = [[xpp_list[j], xhp_list[j]] for j in range(self.cls_p - 1)]

        xp_list = [self.update_conv_list[j](xp_list[j], message_list[j]) for j in range(self.cls_p - 1)]
        att_list = list(torch.split((torch.cat(self_att_list, dim=1) + torch.cat(att_list, dim=1))/2.0, 1, dim=1))
        return xp_list, att_list

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
        xh_list_new, fh_att_list = self.half_infer(h_fea, xh_list, xf, xp_list)
        # for part node
        xp_list_new, hp_att_list = self.part_infer(p_fea, xp_list, xh_list)

        return xp_list_new, xh_list_new, xf_new, hp_att_list, fh_att_list, xf_message


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

        # feature d_transform
        self.p_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_p, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))
        self.h_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_h, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))
        self.f_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_f, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))

        self.relu = nn.ReLU(inplace=False)

        self.bg_conv2 = nn.Sequential(
            nn.Conv2d(3 * in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

        self.gnn2=GNN(adj_matrix, upper_half_node, lower_half_node, self.ch_in, self.hidden, self.cp, self.ch, self.cf)

        self.p_dconv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_p, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))
        self.h_dconv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_h, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))
        self.f_dconv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_f, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))

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



        self.f_message_cls2 = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True,
                                       groups=cls_f)
        self.h_message_cls2 = nn.Conv2d(self.hidden * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True,
                                       groups=cls_h)
        self.p_message_cls2 = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True,
                                       groups=cls_p)

        self.pg_cls_new2 = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.hg_cls_new2 = nn.Conv2d(self.hidden * cls_h, cls_h, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_h)
        self.fg_cls_new2 = nn.Conv2d(self.hidden * cls_f, cls_f, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_f)

    def forward(self, xp, xh, xf):
        # feature transform
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden, dim=1))
        f_node = self.f_conv(xf)
        bg_node = self.bg_conv(torch.cat([xp, xh, xf], dim=1))

        # gnn infer
        p_fea_list_new, h_fea_list_new, f_fea_new, hp_att_list, fh_att_list, f_message = self.gnn(p_node_list, h_node_list, f_node, bg_node, xp, xh, xf)
        xp_infer = self.relu(xp + self.p_dconv(torch.cat([bg_node] + p_fea_list_new, dim=1)))
        xh_infer = self.relu(xh + self.h_dconv(torch.cat([bg_node] + h_fea_list_new, dim=1)))
        xf_infer = self.relu(xf + self.f_dconv(torch.cat([bg_node, f_fea_new], dim=1)))

        bg_node2 = self.bg_conv2(torch.cat([xp_infer, xh_infer, xf_infer], dim=1))

        # gnn infer
        p_fea_list_new2, h_fea_list_new2, f_fea_new2, hp_att_list2, fh_att_list2, f_message2 = self.gnn2(p_fea_list_new, h_fea_list_new, f_fea_new, bg_node2, xp_infer, xh_infer, xf_infer)

        xp_infer2 = self.relu(xp_infer + self.p_dconv2(torch.cat([bg_node2] + p_fea_list_new2, dim=1)))
        xh_infer2 = self.relu(xh_infer + self.h_dconv2(torch.cat([bg_node2] + h_fea_list_new2, dim=1)))
        xf_infer2 = self.relu(xf_infer + self.f_dconv2(torch.cat([bg_node2, f_fea_new2], dim=1)))


        # node supervision
        pg_seg = self.pg_cls(torch.cat([bg_node] + p_node_list, dim=1))
        hg_seg = self.hg_cls(torch.cat([bg_node] + h_node_list, dim=1))
        fg_seg = self.fg_cls(torch.cat([bg_node, f_node], dim=1))

        pg_seg_new = self.pg_cls_new(torch.cat([bg_node] + p_fea_list_new, dim=1))
        hg_seg_new = self.hg_cls_new(torch.cat([bg_node] + h_fea_list_new, dim=1))
        fg_seg_new = self.fg_cls_new(torch.cat([bg_node] + [f_fea_new], dim=1))

        pg_seg_new2 = self.pg_cls_new2(torch.cat([bg_node] + p_fea_list_new2, dim=1))
        hg_seg_new2 = self.hg_cls_new2(torch.cat([bg_node] + h_fea_list_new2, dim=1))
        fg_seg_new2 = self.fg_cls_new2(torch.cat([bg_node] + [f_fea_new2], dim=1))

        pg_seg = sum([pg_seg, pg_seg_new, pg_seg_new2])/3.0
        hg_seg = sum([hg_seg, hg_seg_new, hg_seg_new2])/3.0
        fg_seg = sum([fg_seg, fg_seg_new, fg_seg_new2])/3.0
        fh_att_list = list(torch.split((torch.cat(fh_att_list, dim=1)+torch.cat(fh_att_list2, dim=1))/2.0, 1, dim=1))
        hp_att_list = list(torch.split((torch.cat(hp_att_list, dim=1)+ torch.cat(hp_att_list2, dim=1))/2.0, 1, dim=1))
        return xp_infer2, xh_infer2, xf_infer2, pg_seg, hg_seg, fg_seg, hp_att_list, fh_att_list

class fuse_DecoderModule(nn.Module):

    def __init__(self, num_classes=7, cls_h=3, cls_f=2):
        super(fuse_DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256*3, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True)

        self.conv6 = nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3, xl):
        xt = self.conv0(torch.cat([x1,x2,x3], dim=1))
        xt_fea = self.conv1(xt)

        _, _, th, tw = xl.size()
        xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        x_seg = self.conv4(x_fea+xt)
        h_seg = self.conv5(xt_fea)
        f_seg = self.conv6(xt_fea)
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
        self.fuse_seg = fuse_DecoderModule(num_classes=7, cls_h=3, cls_f=2)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        x_seg, x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb, alpha_hb_fea= self.layerh(seg, x[1])
        alpha_fb, alpha_fb_fea = self.layerf(seg, x[1])

        xp_seg, xh_seg, xf_seg, pg_seg, hg_seg, fg_seg, hp_att_list, fh_att_list  =self.gnn_infer(x_fea, alpha_hb_fea, alpha_fb_fea)

        p_seg, h_seg, f_seg = self.fuse_seg(xp_seg, xh_seg, xf_seg, x[0])

        return [p_seg, h_seg, f_seg, x_seg, alpha_hb, alpha_fb, pg_seg, hg_seg, fg_seg, hp_att_list, fh_att_list, x_dsn]



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
