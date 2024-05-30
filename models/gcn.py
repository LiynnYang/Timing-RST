# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/30 19:37
# @Author      : gwsun
# @Project     : RSMT
# @File        : gcn.py
# @env         : PyCharm
# @Description :

from torch import nn
import torch


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = torch.sum(adj, 1)  # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = torch.pow(rowsum, -0.5)  # (行和rowsum)^(-1/2)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.  # isinf部分赋值为0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, out_channels, bias):
        super().__init__()
        self.W = nn.Linear(in_features, out_features * out_channels, bias=bias)
        self.out_channels = out_channels
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

    def forward(self, input, adj, mask=None):
        batch_size = input.shape[0]
        degree = input.shape[1]
        adj_normal = normalize_adj(adj.to(torch.float32)).unsqueeze(1)
        h = self.W(input).view(batch_size, degree, self.out_channels, -1).transpose(1, 2)
        return torch.matmul(adj_normal, h).mean(1)


class GCNLayerV2(nn.Module):
    def __init__(self, in_features, out_features, out_channels, bias=False):
        super(GCNLayerV2, self).__init__()
        self.conv1 = GCNLayer(in_features, out_features, out_channels, bias)
        self.conv2 = GCNLayer(in_features, out_features, out_channels, bias)
        self.conv3 = GCNLayer(in_features, out_features, out_channels, bias)

    def forward(self, input, adj, adj_in, adj_out):
        batch_size = adj.shape[0]
        degree = adj.shape[1]
        no_dir_adj = adj.transpose(1, 2).to(torch.int8) | adj.to(torch.int8)
        h1 = self.conv1(input, no_dir_adj.to(torch.float32))
        h2 = self.conv2(input, adj_in)
        h3 = self.conv3(input, adj_out)
        return torch.cat([h1, h2, h3], dim=-1).reshape(batch_size, degree, 3, -1).mean(2)


if __name__ == '__main__':
    adj = torch.zeros([3, 5, 5])
    normalize_adj(adj)
