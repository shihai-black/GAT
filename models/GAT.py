# -*- coding: utf-8 -*-
# @project：hot_item_mining
# @author:caojinlei
# @file: GAT.py
# @time: 2022/05/24
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 input_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        if num_layers > 1:
            self.gat_layers.append(GATConv(
                input_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers - 1):
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l - 1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # outut layer
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                input_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None
            ))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            h = layer(block, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h