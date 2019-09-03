import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

from modules.dcn import DFConv2d

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
        _, _, th, tw = xl.size()
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

class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1,2,3,4], lower_part_list=[5,6], in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]


        self.part_dp_list = nn.ModuleList([Part_Dependency(hidden_dim) for i in range(self.edge_index_num)])

        self.update_conv_list = nn.ModuleList(
            [nn.Sequential(
            nn.Conv2d(2*hidden_dim, 32, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(cls_p-1)])


    def forward(self, xp_list):
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

        message_list = xpp_list

        xp_list = [
            self.update_conv_list[j](torch.cat([xp_list[j], message_list[j]], dim=1))
            for j in range(self.cls_p - 1)]
        return xp_list, message_list


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
            ),BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

        self.A_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())
        self.B_att = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, pA, pB):
        A_diffuse = self.dconv(torch.cat([pA, pB], dim=1))
        A_att = self.A_att(pA)
        A_diffuse_att = (2 - A_att) * A_diffuse
        return A_diffuse_att


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

        self.part_infer = Part_Graph(adj_matrix,self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h, cls_f)


    def forward(self, xp_list):
        # for part node
        xp_list_new, xp_message_list = self.part_infer(xp_list)

        return xp_list_new, xp_message_list


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
        self.bg_conv = nn.Sequential(
            nn.Conv2d(3*in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn=GNN(adj_matrix, upper_half_node, lower_half_node, self.ch_in, self.hidden, self.cp, self.ch, self.cf)

        # feature d_transform
        self.p_dconv = nn.Sequential(
            nn.Conv2d(hidden_dim * cls_p, in_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(in_dim))

        self.relu = nn.ReLU(inplace=False)

        #node supervision
        # classifier
        self.pg_cls = nn.Conv2d(self.hidden*cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        # self.p_message_cls = nn.Conv2d(self.hidden*cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)
        self.pg_cls_new = nn.Conv2d(self.hidden * cls_p, cls_p, kernel_size=1, padding=0, stride=1, bias=True, groups=cls_p)


    def forward(self, xp, xh, xf):
        # feature transform
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden, dim=1))
        bg_node = self.bg_conv(torch.cat([xp, xh, xf], dim=1))

        # gnn infer
        p_fea_list_new, p_message_list = self.gnn(p_node_list)
        xp_infer = torch.cat([xp]+ [bg_node] + p_fea_list_new, dim=1)

        #node supervision
        pg_seg = self.pg_cls(torch.cat([bg_node]+p_node_list, dim=1))
        pg_seg_new = self.pg_cls_new(torch.cat([bg_node]+p_fea_list_new, dim=1))

        #message supervision
        # p_message_seg = self.p_message_cls(torch.cat([bg_node]+p_message_list, dim=1))
        pg_seg = sum([pg_seg, pg_seg_new])

        return xp_infer, pg_seg

class fuse_DecoderModule(nn.Module):

    def __init__(self, hidden_dim=20, num_classes=7, cls_h=3, cls_f=2):
        super(fuse_DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256+num_classes*hidden_dim, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

        self.conv5 = nn.Sequential(nn.Conv2d(256, cls_h, kernel_size=1, padding=0, dilation=1, bias=True))

        self.conv6 = nn.Sequential(nn.Conv2d(256, cls_f, kernel_size=1, padding=0, dilation=1, bias=True))


    def forward(self, x1, x2, x3, xl):
        _, _, th, tw = xl.size()
        xt = self.conv0(F.interpolate(x1, size=(th, tw), mode='bilinear', align_corners=True))
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        x_seg = self.conv4(x_fea)
        h_seg = self.conv5(x2)
        f_seg = self.conv6(x3)
        return x_seg, h_seg, f_seg

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = MagicModule(2048, 512, 1)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)

        self.adj_matrix=torch.tensor([[0,1,0,0,0,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,1],[0,0,0,0,1,0]], requires_grad=False)
        self.gnn_infer=GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1,2,3,4], lower_half_node=[5,6], in_dim=256, hidden_dim=40, cls_p=7, cls_h=3, cls_f=2)
        self.fuse_seg = fuse_DecoderModule(hidden_dim=40, num_classes=7, cls_h=3, cls_f=2)
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
        xp_seg, node_p_seg =self.gnn_infer(x_fea,alpha_hb_fea,alpha_fb_fea)
        p_seg, h_seg, f_seg = self.fuse_seg(xp_seg, alpha_hb_fea, alpha_fb_fea, x[0])
        return p_seg, h_seg, f_seg, node_p_seg, x_dsn


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
