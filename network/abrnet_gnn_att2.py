import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154
from modules.dcn import DFConv2d
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

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
    def __init__(self, in_dim=256*3, hidden_dim=256):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_update = nn.Sequential(DFConv2d(
                in_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False),
            DFConv2d(
                hidden_dim,
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
        self.relu = nn.LeakyReLU()

    def forward(self, x, message_list):
        if len(message_list)>1:
            out = self.conv_update(torch.cat([x]+message_list, dim=1))
        else:
            out = self.conv_update(torch.cat([x, message_list[0]], dim=1))
        return self.relu(x+self.gamma*out)


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
        self.conv_Update = conv_Update(in_dim*3, in_dim)

    def forward(self, xp_fea,xh_fea, xf_fea, xf, f_att):
        xf = self.conv_Update(xf+xf_fea, [f_att*xh_fea, f_att*xp_fea])
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

        self.update_u = conv_Update(in_dim*3, in_dim)
        self.update_l = conv_Update(in_dim*3, in_dim)

    def forward(self, xp_fea,xh_fea, xf_fea, xh_list, h_att_list):
        # upper half
        message_u = [h_att_list[0]*xf_fea, h_att_list[0]*xp_fea]
        xh_u = self.update_u(xh_list[0]+xh_fea, message_u)

        # lower half
        message_l = [h_att_list[1]*xf_fea, h_att_list[1]*xp_fea]
        xh_l = self.update_l(xh_list[1]+xh_fea, message_l)

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

        # self.part_dp_list = nn.ModuleList([Part_Dependency(in_dim, hidden_dim) for i in range(self.edge_index_num)])
        # self.att = nn.Sequential(
        #     nn.Conv2d((cls_p - 1) * hidden_dim, cls_p - 1, kernel_size=1, padding=0, stride=1, bias=True,
        #               groups=cls_p - 1),
        #     nn.Sigmoid())

        self.update_conv_list = nn.ModuleList(
            [conv_Update(in_dim*3, in_dim) for i in range(cls_p - 1)])

    def forward(self, xp_fea,xh_fea, xf_fea, xp_list, p_att_list):

        # dp_att_list = torch.split(self.att(torch.cat(xp_list, dim=1)), 1, dim=1)
        # xpp_list_list = [[] for i in range(self.cls_p - 1)]
        # xpp_list = []
        # for i in range(self.edge_index_num):
        #     xpp_list_list[self.edge_index[i, 1]].append(
        #         self.part_dp_list[i](p_fea, xp_list[self.edge_index[i, 0]], xp_list[self.edge_index[i, 1]],
        #                              dp_att_list[self.edge_index[i, 0]], dp_att_list[self.edge_index[i, 1]]))
        #
        # for i in range(self.cls_p - 1):
        #     if len(xpp_list_list[i]) == 1:
        #         xpp_list.append(xpp_list_list[i][0])
        #     else:
        #         xpp_list.append(sum(xpp_list_list[i]))

        xp_list_new = []
        for i in range(self.cls_p-1):
            if i+1 in self.upper_part_list:
                xp_list_new.append(self.update_conv_list[i](xp_list[i]+xp_fea, [p_att_list[i]*xf_fea, p_att_list[i]*xh_fea]))
            elif i+1 in self.lower_part_list:
                xp_list_new.append(self.update_conv_list[i](xp_list[i]+xp_fea, [p_att_list[i]*xf_fea, p_att_list[i]*xh_fea]))
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

    def forward(self, xp_fea,xh_fea,xf_fea, xp_list, xh_list, xf, p_att_list, h_att_list, f_att):
        # for full body node
        xf_new = self.full_infer(xp_fea,xh_fea,xf_fea, xf, f_att)
        # for half body node
        xh_list_new = self.half_infer(xp_fea,xh_fea,xf_fea, xh_list, h_att_list)
        # for part node
        xp_list_new = self.part_infer(xp_fea,xh_fea,xf_fea, xp_list, p_att_list)

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
        self.bg_conv_new = nn.Sequential(
            nn.Conv2d((cls_p + cls_h + cls_f - 2) * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1,
                      bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn = GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p,
                       self.cls_h, self.cls_f)

        # node supervision
        self.pg_cls = nn.Conv2d(in_dim*4, cls_p, kernel_size=1, padding=0, stride=1, bias=True,
                                groups=1)
        self.hg_cls = nn.Conv2d(in_dim*4, cls_h, kernel_size=1, padding=0, stride=1, bias=True,
                                groups=1)
        self.fg_cls = nn.Conv2d(in_dim*4, cls_f, kernel_size=1, padding=0, stride=1, bias=True,
                                groups=1)

        # self.final_cls = Final_classifer(in_dim, hidden_dim, cls_p, cls_h, cls_f)

        self.p_att = nn.Sequential(
            nn.Conv2d(in_dim, (cls_p), kernel_size=1, padding=0, stride=1, bias=True, groups=1),
            nn.Sigmoid())
        self.h_att = nn.Sequential(
            nn.Conv2d(in_dim, (cls_h), kernel_size=1, padding=0, stride=1, bias=True, groups=1),
            nn.Sigmoid())
        self.f_att = nn.Sequential(
            nn.Conv2d(in_dim, (cls_f), kernel_size=1, padding=0, stride=1, bias=True, groups=1),
            nn.Sigmoid())

    def forward(self, xp, xh, xf):
        # _, _, th, tw = xp.size()
        # _, _, h, w = xh.size()
        #
        # xh = F.interpolate(xh, (th, tw), mode='bilinear', align_corners=True)
        # xf = F.interpolate(xf, (th, tw), mode='bilinear', align_corners=True)

        f_att = self.f_att(xf)
        h_att = self.h_att(xh)
        p_att = self.p_att(xp)
        f_att_list = list(torch.split(f_att, 1, dim=1))
        h_att_list = list(torch.split(h_att, 1, dim=1))
        p_att_list = list(torch.split(p_att, 1, dim=1))
        # feature transform
        f_node = f_att_list[1]*xf
        p_node_list = [p_att_list[i]*xp for i in range(1, self.cls_p)]
        h_node_list = [h_att_list[i]*xp for i in range(1, self.cls_h)]
        bg_node_f = f_att_list[0]*xf
        bg_node_h = h_att_list[0]*xh
        bg_node_p = p_att_list[0]*xp

        # gnn infer
        p_node_list_new, h_node_list_new, f_node_new = self.gnn(xp,xh,xf,p_node_list, h_node_list, f_node, p_att_list[1:], h_att_list[1:], f_att_list[1])

        xphf_infer = torch.cat([sum(p_node_list_new)+bg_node_p, sum(h_node_list_new)+bg_node_h, f_node_new+bg_node_f], dim=1)
        p_seg = self.pg_cls(torch.cat([xphf_infer, xp], dim=1))
        h_seg = self.hg_cls(torch.cat([xphf_infer, xh], dim=1))
        f_seg = self.fg_cls(torch.cat([xphf_infer, xf], dim=1))
        # p_seg_final, h_seg_final, f_seg_final = self.final_cls(xphf_infer, xp, xh, xf)

        return p_seg, h_seg, f_seg, p_att, h_att, f_att


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
                                   in_dim=256, hidden_dim=40, cls_p=7, cls_h=3, cls_f=2)
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
        p_seg, h_seg, f_seg, p_att, h_att, f_att = self.gnn_infer(x_fea, alpha_hb_fea, alpha_fb_fea)
        return p_seg, h_seg, f_seg, p_att, h_att, f_att, x_dsn


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
