# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/11 20:32
# @Author      : gwsun
# @Project     : RSMT
# @File        : gat.py
# @env         : PyCharm
# @Description :
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, temperature=1, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout
        self.num_heads = num_heads
        self.out_featuers = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2 * out_features, num_heads)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, input, adj, mask=None):
        """

        :param input: [batch_size * degree * in_features]
        :param adj: [batch_size * degree * degree]
        :param mask:
        :return:
        """
        batch_size = input.shape[0]
        degree = input.shape[1]
        input = self.W(input)  # [batch_size * degree * out_features]
        input1 = input.repeat(1, 1, degree).reshape(batch_size, degree * degree, -1)
        input2 = input.repeat(1, degree, 1)
        final_input = torch.cat([input1, input2], dim=-1)  # [batch_size, degree*degree, 2out_features]
        e = self.leakyRelu(self.a(final_input).transpose(1, 2)).view(batch_size, self.num_heads, degree, degree)  # [batch_size, num_heads, degree*degree]

        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [batch_size, num_heads, degree, degree]
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # [batch_size, num_heads, degree, degree]
        attention = F.dropout(attention, self.dropout, training=self.training)  # 将一部分元素置为0，其他元素会乘以 scale 1/(1-p).
        h_prime = torch.matmul(attention, input.unsqueeze(1)).mean(1)
        # h_prime = torch.matmul(adj.to(torch.float32), input.unsqueeze(1)).mean(1)
        return h_prime


class TripleGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super().__init__()
        self.conv1 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        self.conv2 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        self.conv3 = GATLayer(in_feats, out_feats, num_heads=num_heads)
        # self.W = nn.Linear(3 * out_feats, out_feats)

    def forward(self, input, adj, adj_in, adj_out):
        batch_size = adj.shape[0]
        degree = adj.shape[1]
        h1 = self.conv1(input, adj)
        h2 = self.conv2(input, adj_in)
        h3 = self.conv3(input, adj_out)
        return torch.cat([h1, h2, h3], dim=-1).reshape(batch_size, degree, 3, -1).mean(2)
        # return self.W(torch.cat([h1, h2, h3], dim=-1))


class TripleGATLayerV2(nn.Module):
    def __init__(self, in_features, out_features, num_heads, temperature=1, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout
        self.num_heads = num_heads
        self.out_featuers = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2 * out_features, num_heads*3)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, input, adj, adj_in, adj_out):
        batch_size = input.shape[0]
        degree = input.shape[1]
        input = self.W(input)  # [batch_size * degree * out_features]
        input1 = input.repeat(1, 1, degree).reshape(batch_size, degree * degree, -1)
        input2 = input.repeat(1, degree, 1)
        final_input = torch.cat([input1, input2], dim=-1)  # [batch_size, degree*degree, 2out_features]
        e = self.leakyRelu(self.a(final_input).transpose(1, 2)).view(batch_size, self.num_heads*3, degree,
                                                                     degree)  # [batch_size, num_heads, degree*degree]

        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [batch_size, num_heads, degree, degree]
        adj_in = adj_in.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        adj_out = adj_out.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        adj_final = torch.cat([adj, adj_in, adj_out], 1)
        attention = torch.where(adj_final > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # [batch_size, num_heads, degree, degree]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, input.unsqueeze(1)).mean(1)
        # h_prime = torch.matmul(adj.to(torch.float32), input.unsqueeze(1)).mean(1)
        return h_prime
        # return torch.cat([h1, h2, h3, h4], dim=-1).reshape(batch_size, degree, 4, -1).mean(2)


class LocalGATFirstOrder(nn.Module):
    def __init__(self, feats, num_heads):
        super().__init__()
        self.gat_layer = TripleGATLayer(feats, feats, num_heads)

    def forward(self, input, adj, adj_in, adj_out, edges):
        """
        :param input:
        :param adj:
        :param edges: 输入形式[[x1, x2, x3, ..., xb], [y1, y2, y3, ...,yb]]
        :return:
        """
        batch_size = adj.shape[0]
        degree = adj.shape[1]
        feats = input.shape[-1]
        # TODO： 需处理选点0，0的图，代表已经结束，则不再更新（其实更新也无所谓，反正后续不会再对它处理了,暂未处理）
        with torch.no_grad():
            adj_nodir = adj | adj.transpose(1, 2)
            adj_part = adj_nodir[list(range(batch_size)), edges].transpose(0, 1)
            adj_points = adj_part[:, 0] | adj_part[:, 1]
            valid_len = adj_points.sum(-1)
            max_len = valid_len.max()
            index = torch.where(adj_points == 1)  # 记录真实的邻接点 e.g.(tensor([0, 0, 0, 1, 1]), tensor([0, 1, 2, 1, 2]))
            point_split = index[1].split(valid_len.tolist())
            new_points = pad_sequence(point_split, padding_value=degree-1)  # 填充0则会每次都将0引入  [[0, 1, 2],
                                                                                                              #[1, 2, 0]]
            # new_points = new_points.sort(1, False)[0]
            adj_new = adj[list(range(batch_size)), new_points].transpose(0, 1)
            adj_new = adj_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)

            adj_in_new = adj_in[list(range(batch_size)), new_points].transpose(0, 1)
            adj_in_new = adj_in_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)

            adj_out_new = adj_out[list(range(batch_size)), new_points].transpose(0, 1)
            adj_out_new = adj_out_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)
            # 现在的新矩阵为多余的行和列都是与点0的邻接关系
            # 求邻接矩阵的mask
            # mask_len = max_len - valid_len
            # mask_len = mask_len.unsqueeze(1).unsqueeze(1)
            # valid_len = valid_len.unsqueeze(1).unsqueeze(1)
            # mask = torch.arange(max_len, device=valid_len.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1)
            # mask = mask.lt(valid_len)
            # mask = mask & mask.transpose(1, 2)
            # adj_new = adj_new & mask
            # adj_in_new = adj_in_new & mask
            # adj_out_new = adj_out_new & mask
            # adj_new[:, list(range(max_len)), list(range(max_len))] = 1
            # adj_in_new[:, list(range(max_len)), list(range(max_len))] = 1
            # adj_out_new[:, list(range(max_len)), list(range(max_len))] = 1

        # new_input = input.clone()
        input_part = input[list(range(batch_size)), new_points].transpose(0, 1)
        new_input_part = self.gat_layer(input_part, adj_new, adj_in_new, adj_out_new)
        input[list(range(batch_size)), new_points] = new_input_part.transpose(0, 1)
        # return new_input


class LocalGATSecondOrder(nn.Module):
    def __init__(self, feats, num_heads):
        super().__init__()
        self.gat_layer = TripleGATLayer(feats, feats, num_heads)

    def forward(self, input, adj, adj_in, adj_out, edges):
        """
        :param input:
        :param adj:
        :param edges: 输入形式[[x1, x2, x3, ..., xb], [y1, y2, y3, ...,yb]]
        :return:
        """
        batch_size = adj.shape[0]
        degree = adj.shape[1]
        with torch.no_grad():
            adj_nodir = adj | adj.transpose(1, 2)
            adj_part_1 = adj_nodir[list(range(batch_size)), edges].transpose(0, 1)
            adj_points_1 = adj_part_1[:, 0] | adj_part_1[:, 1]
            valid_len_1 = adj_points_1.sum(-1)
            max_len_1 = valid_len_1.max()
            index_1 = torch.where(adj_points_1 == 1)  # 记录真实的邻接点 e.g.(tensor([0, 0, 0, 1, 1]), tensor([0, 1, 2, 1, 2]))
            point_split_1 = index_1[1].split(valid_len_1.tolist())
            new_points_1 = pad_sequence(point_split_1, padding_value=degree-1)  # 填充0则会每次都将0引入  [[0, 1, 2],
                                                                                                              #[1, 2, 0]]
            # TODO：根据一阶邻接点再获取到二阶邻接点
            adj_part_2 = adj_nodir[list(range(batch_size)), new_points_1].transpose(0, 1)  # batch_size * max_len * degree
            # tmp = torch.arange(max_len_1, device=valid_len_1.device).unsqueeze(1).repeat(1, degree).unsqueeze(0).repeat(batch_size, 1, 1)
            # mask_row = tmp.lt(valid_len_1.unsqueeze(1).unsqueeze(1))
            # adj_part_2 = adj_part_2 & mask_row
            adj_points = torch.where(adj_part_2.sum(1) >= 1, 1, 0)
            valid_len = adj_points.sum(-1)
            max_len=valid_len.max()
            index = torch.where(adj_points == 1)
            point_split = index[1].split(valid_len.tolist())
            new_points = pad_sequence(point_split, padding_value=degree-1)
            # new_points = new_points.sort(1, False)[0]

            adj_new = adj[list(range(batch_size)), new_points].transpose(0, 1)
            adj_new = adj_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)

            adj_in_new = adj_in[list(range(batch_size)), new_points].transpose(0, 1)
            adj_in_new = adj_in_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)

            adj_out_new = adj_out[list(range(batch_size)), new_points].transpose(0, 1)
            adj_out_new = adj_out_new[list(range(batch_size)), :, new_points].permute(1, 2, 0)
            # 现在的新矩阵为多余的行和列都是与点0的邻接关系

            # mask_len = max_len - valid_len
            # mask_len = mask_len.unsqueeze(1).unsqueeze(1)
            # mask = torch.arange(max_len, device=valid_len.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1)
            # mask = mask.ge(mask_len)
            # mask = mask & mask.transpose(1, 2)
            # adj_new = adj_new & mask
            # adj_in_new = adj_in_new & mask
            # adj_out_new = adj_out_new & mask
            # adj_new[:, list(range(max_len)), list(range(max_len))] = 1
            # adj_in_new[:, list(range(max_len)), list(range(max_len))] = 1
            # adj_out_new[:, list(range(max_len)), list(range(max_len))] = 1

        # new_input = input.clone()
        input_part = input[list(range(batch_size)), new_points].transpose(0, 1)
        new_input_part = self.gat_layer(input_part, adj_new, adj_in_new, adj_out_new)
        input[list(range(batch_size)), new_points] = new_input_part.transpose(0, 1)
        # return new_input


class LocalGAT(nn.Module):
    def __init__(self, feats, num_heads=5):
        super().__init__()
        self.firstOrderGat = LocalGATFirstOrder(feats, num_heads)
        self.secondOrderGat = LocalGATSecondOrder(feats, num_heads)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(feats)
        self.bn2 = nn.BatchNorm1d(feats)
        self.feats = feats
    def forward(self, input, adj, adj_in, adj_out, edges):
        batch = input.shape[0]
        degree = input.shape[1]
        device = input.device
        new_input = torch.zeros([batch, degree + 1, self.feats], device=device)
        new_input[:, :-1, :] = input.clone()
        with torch.no_grad():
            new_adj = torch.zeros([batch, degree + 1, degree+1], dtype=torch.int64, device=device)
            new_adj_in = torch.zeros([batch, degree + 1, degree + 1], dtype=torch.int64, device=device)
            new_adj_out = torch.zeros([batch, degree + 1, degree + 1], dtype=torch.int64, device=device)
            new_adj[:, :-1, :-1] = adj
            new_adj_in[:, :-1, :-1] = adj_in
            new_adj_out[:, :-1, :-1] = adj_out
            # new_adj[:, -1, -1] = 1
            # new_adj_in[:, -1, -1] = 1
            # new_adj_out[:, -1, -1] = 1
        self.firstOrderGat(new_input, new_adj, new_adj_in, new_adj_out, edges)
        # new_input[:, :-1, :] = self.bn1(new_input[:, :-1, :].reshape(batch * degree, -1)).reshape(batch, degree, -1)
        new_input = self.relu(new_input)
        new_input = new_input.clone()
        # node_embedding = F.layer_norm(node_embedding, [degree, self.hid_feats])
        # node_embedding = self.relu(node_embedding)
        self.secondOrderGat(new_input, new_adj, new_adj_in, new_adj_out, edges)
        # new_input[:, :-1, :] = self.bn2(new_input[:, :-1, :].reshape(batch * degree, -1)).reshape(batch, degree, -1)
        new_input = new_input[:, :-1, :]
        new_input = F.layer_norm(new_input, [degree, self.feats])

        new_input = self.relu(new_input)

        return new_input


if __name__ == '__main__':
    # input = torch.from_numpy(np.random.rand(100, 5, 128)).to(torch.float32)
    # adj = torch.tensor([[1, 0, 1, 0, 0],
    #                     [0, 1, 1, 0, 1],
    #                     [1, 0, 1, 1, 0],
    #                     [0, 1, 1, 1, 0],
    #                     [1, 1, 0, 0, 1]])
    # adj = adj.unsqueeze(0).repeat(100, 1, 1)
    # gat = GATLayer(128, 256, 5, 1)
    # h = gat(input, adj)
    # print(h.shape)
    # print()
    adj = torch.zeros([2, 8, 8], dtype=torch.int64)
    adj[[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 5, 6], [1, 5, 3, 5, 6, 4]] = 1
    adj[[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 2, 2, 2, 3], [1, 4, 6, 0, 3, 5, 7]] = 1
    adj[:, list(range(8)), list(range(8))] = 1
    print(adj)
    # first = LocalGATFirstOrder(10, 10, 5)
    local_gat = LocalGAT(10)
    input = torch.rand(2, 8, 10)
    # degree = 8, features=10, batch_size = 2
    new = local_gat(input, adj, adj, adj, [[0, 0], [1, 1]])
    print()
