# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2023/4/18 19:59
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : model_sorted.py
# @env         : PyCharm
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions.categorical import Categorical
from models.utils import Embedder, Pointer, Glimpse
from models.self_attn import Encoder
from models.gcn import GCNLayerV2
from models.gat import TripleGATLayer
"""
generate the edge on graph from start point one by one
"""

class TripleGAT(nn.Module):
    # TODO:注意力机制不对称的
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = TripleGATLayer(in_feats, hid_feats, 5)
        self.conv2 = TripleGATLayer(hid_feats, out_feats, 5)
        # self.bn1 = nn.BatchNorm1d(hid_feats)  # 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作
        # self.bn2 = nn.BatchNorm1d(out_feats)

    def forward(self, adj, adj_in, adj_out, inputs):
        # 输入是节点的特征
        batch = adj.shape[0]
        degree = adj.shape[-1]

        h = self.conv1(inputs, adj, adj_in, adj_out)
        # h = self.bn1(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        h = F.relu(h)
        # h = F.leaky_relu(h)
        # h = F.tanh(h)
        h = self.conv2(h, adj, adj_in, adj_out)
        # h = self.bn2(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        return h

class TripleGNN(nn.Module):
    # TODO:注意力机制不对称的
    def __init__(self, in_feats, hid_feats, out_feats, channels=5):
        super().__init__()
        self.conv1 = GCNLayerV2(in_feats, hid_feats, channels)
        self.conv2 = GCNLayerV2(hid_feats, out_feats, channels)
        # self.bn1 = nn.BatchNorm1d(hid_feats)  # 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作
        # self.bn2 = nn.BatchNorm1d(out_feats)

    def forward(self, adj, adj_in, adj_out, inputs):
        # 输入是节点的特征
        batch = adj.shape[0]
        degree = adj.shape[-1]

        h = self.conv1(inputs, adj, adj_in, adj_out)
        # h = self.bn1(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        h = F.relu(h)
        # h = F.tanh(h)
        h = self.conv2(h, adj, adj_in, adj_out)
        # h = self.bn2(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        h = F.relu(h)
        return h


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Tanh(),
        )
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
        self.bn = nn.BatchNorm1d(self.graph_hidden_size)

    def forward(self, input):
        graph_emb = (self.node_gating(input) * self.node_to_graph(input)).sum(1)
        graph_emb = self.bn(graph_emb)
        return graph_emb


# class GraphEmbed2(nn.Module):
#     def __init__(self, node_hidden_size):
#         super(GraphEmbed2, self).__init__()
#
#         # Setting from the paper
#         self.graph_hidden_size = 2 * node_hidden_size
#         self.node_gating_visited = nn.Sequential(
#             nn.Linear(node_hidden_size, 1),
#             nn.Tanh(),
#         )
#         self.node_gating_unvisited = nn.Sequential(
#             nn.Linear(node_hidden_size, 1),
#             nn.Tanh(),
#         )
#         self.visited_node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
#         self.unvisited_node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
#         self.bn = nn.BatchNorm1d(self.graph_hidden_size)
#         self.w1 = nn.Linear(self.graph_hidden_size, self.graph_hidden_size, bias=False)
#         self.w2 = nn.Linear(self.graph_hidden_size, self.graph_hidden_size, bias=False)
#
#     def forward(self, input, visited):
#         visited_input = torch.masked_select(input,  (visited.unsqueeze(-1) == 1)).reshape(input.shape[0], -1, input.shape[-1])
#         unvisited_input = torch.masked_select(input,  (visited.unsqueeze(-1) == 0)).reshape(input.shape[0], -1, input.shape[-1])
#         graph_emb_visited = (self.node_gating_visited(visited_input) * self.visited_node_to_graph(visited_input)).sum(1)
#         graph_emb_unvisited = (self.node_gating_unvisited(unvisited_input) * self.unvisited_node_to_graph(unvisited_input)).sum(1)
#         graph_emb = self.bn(self.w1(graph_emb_visited) + self.w2(graph_emb_unvisited))
#         return graph_emb


class GraphEmbed3(nn.Module):
    def __init__(self, d_model, d_unit):
        super(GraphEmbed3, self).__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.glimpse_unvisited = Glimpse(d_model, d_unit)
        self.glimpse_visited = Glimpse(d_model, d_unit)
        self.relu = nn.ReLU()

    def forward(self, input, visited, graph_mask=None):
        visited_input = torch.masked_select(input, (visited.unsqueeze(-1) == 1)).reshape(input.shape[0], -1,
                                                                                         input.shape[-1])
        unvisited_input = torch.masked_select(input, (visited.unsqueeze(-1) == 0)).reshape(input.shape[0], -1,
                                                                                           input.shape[-1])

        graph_mask1, graph_mask2 = None, None
        if graph_mask is not None:
            graph_mask1 = torch.masked_select(graph_mask, visited == 1).reshape(visited_input.shape[0], -1,
                                                                                visited_input.shape[-1])
            graph_mask2 = torch.masked_select(graph_mask, visited == 0).reshape(unvisited_input.shape[0], -1,
                                                                                unvisited_input.shape[-1])
        g_emb_visited = self.glimpse_visited(visited_input, graph_mask1)
        g_emb_unvisited = self.glimpse_unvisited(unvisited_input, graph_mask2)
        graph_emb = self.w1(g_emb_unvisited) + self.w2(g_emb_visited)
        graph_emb = self.relu(graph_emb)
        return graph_emb


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.device = None
        # embedder args
        self.d_input = 3
        self.d_model = 128
        self.embedder = Embedder(self.d_input, self.d_model)

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        # self.seq = nn.Sequential(nn.Linear(self.d_model, 2 * self.d_model),
        #                          nn.ReLU(),
        #                          nn.Linear(2*self.d_model, self.d_model))
        # feedforward layer inner
        self.d_inner = 512
        self.d_unit = 256
        # self.pos_ffn = PositionwiseFeedForward(self.d_model, self.d_inner)
        self.encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)

        # decoder args
        # self.ptr = Pointer(2 * self.d_model, self.d_model)
        self.ptr = Pointer(self.d_model, self.d_model)
        # TODO: 需要考虑用于初始化的GAT是否需要与用于更新的GAT共享一个权重？
        self.gat = TripleGAT(self.d_model, self.d_model, self.d_model)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.BatchNorm1d(self.d_model)
        )
        # self.graph_emb = GraphEmbed2(self.d_model)
        self.graph_emb = GraphEmbed3(self.d_model, self.d_unit)
        # self.glimpse = Glimpse(self.d_model, self.d_unit)  # 若使用这个就要降ptr第一个参数设为d_model
        # self.bn = nn.BatchNorm1d(self.d_model)

    def forward(self, inputs: torch.tensor, deterministic: bool = False, pad_len=None):
        """

        :param inputs: numpy.ndarray [batch_size * degree * 2]
        :param deterministic:
        :return:
        """
        batch_size = inputs.shape[0]
        degree = inputs.shape[1]
        time_start = time.time()
        # 获取当前进程模型所在位置设备
        self.device = inputs.device

        adj = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_in = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_out = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        visited = torch.zeros([batch_size, degree], dtype=torch.int64, device=self.device)
        visited[:, 0] = 1

        indexes, log_probs = [], []
        #为inputs添加第三维标记
        pad = torch.zeros([batch_size, degree, 1], device=self.device)
        pad[:, 0, 0] = 1
        inputs = torch.cat([inputs, pad], -1)
        embedings = self.embedder(inputs)
        # encodings = self.seq(embedings)
        encodings = self.encoder(embedings, None)
        time_emb, time_update_graph, time_update_mask = .0, .0, .0
        # id_node_start = torch.arange(batch_size, dtype=torch.int64, device=self.device) * degree
        graph_mask = None
        if pad_len is not None:
            graph_mask = torch.arange(degree, device=inputs.device).repeat(batch_size, 1)
            tmp = pad_len.reshape(batch_size, -1)
            graph_mask = graph_mask.ge(degree - tmp)
        step = None
        for step in range(degree - 1):
            with torch.no_grad():
                visited_rep = visited.repeat(1, degree).reshape(batch_size, degree, degree)
                mask_check = (visited_rep | visited_rep.transpose(1, 2)) - (visited_rep & visited_rep.transpose(1, 2))
                # batch_size * degree * degree
                if pad_len is not None:
                    for id, pd in enumerate(pad_len):
                        if pd != 0:
                            mask_check[id][-pd:, :] = 0
                            mask_check[id][:, -pd:] = 0
            if torch.all(mask_check == 0):
                break
                # mask_check = torch.ones([batch_size, degree, degree], device=self.device) - \
                #              torch.eye(degree, device=self.device).repeat(batch_size, 1, 1)
            time_start2 = time.time()
            node_embedding = self.gat(adj, adj_in, adj_out, encodings)
            input1 = node_embedding.repeat(1, 1, degree).reshape(batch_size, degree * degree, -1)
            input2 = node_embedding.repeat(1, degree, 1)
            final_input = torch.cat([input1, input2], dim=-1).reshape(batch_size * degree * degree, -1)
            edge_embedding = self.fc(final_input).reshape(batch_size, degree * degree, -1)
            node_embedding = node_embedding.reshape(batch_size, degree, -1)
            # graph_embedding = self.glimpse(node_embedding, graph_mask)
            graph_embedding = self.graph_emb(node_embedding, visited, graph_mask)
            logits, t = self.ptr(edge_embedding, graph_embedding, mask_check)
            time_emb += float(time.time() - time_start2)
            distr = Categorical(logits=logits)
            if deterministic:
                _, edge_idx = torch.max(logits, -1)
            else:
                edge_idx = distr.sample()
            time_start3 = time.time()
            with torch.no_grad():
                x_idx = torch.div(edge_idx, degree, rounding_mode='floor')
                y_idx = torch.fmod(edge_idx, degree)
                # 在更新前先将mask pad部分全置为1，代表此点尚孤立; 更新后即复位为0，代表此点相关边不可选
                # if pad_len is not None:
                #     for id, pd in enumerate(pad_len):
                #         if pd != 0:
                #             mask_check[id][-pd:, :] = 1
                #             mask_check[id][:, -pd:] = 1
                # mask_check = self.update_mask(mask_check, x_idx, y_idx)
                # if pad_len is not None:
                #     for id, pd in enumerate(pad_len):
                #         if pd != 0:
                #             mask_check[id][-pd:, :] = 0
                #             mask_check[id][:, -pd:] = 0
            time_update_mask += float(time.time() - time_start3)
            time_start4 = time.time()
            # 若某个图的连边已经结束，则让其默认取边（0，0）
            indexes.append(x_idx)
            indexes.append(y_idx)
            log_p = distr.log_prob(edge_idx)
            log_p[t] = log_p[t].detach()
            log_probs.append(log_p)
            # 更新visited
            visited.scatter_(1, torch.stack([x_idx, y_idx], 1), 1)
            adj = adj.index_put((torch.arange(batch_size, dtype=torch.int64, device=adj.device), x_idx, y_idx),
                                torch.LongTensor([1]).to(adj.device))

            tmp = adj[torch.arange(batch_size), x_idx]
            tmp[torch.arange(batch_size), x_idx] = 0
            tmp = tmp.unsqueeze(1).repeat(1, degree, 1)
            tmp = tmp * tmp.transpose(2, 1)
            adj_in = adj_in | tmp

            tmp = adj.transpose(2, 1)[torch.arange(batch_size), y_idx]
            tmp[torch.arange(batch_size), y_idx] = 0
            tmp = tmp.unsqueeze(1).repeat(1, degree, 1)
            tmp = tmp * tmp.transpose(2, 1)
            adj_out = adj_out | tmp

            time_update_graph += float(time.time() - time_start4)
            # if single:  # 如果是找环过程，直接运行一次就返回
            #     return x_idx, y_idx
        # log_probs  9*1024
        log_probs = sum(log_probs)
        time_end = float(time.time() - time_start)
        # 返回形式：index：x1, y1, x2, y2, ..., 0, 0  (0,0)是为了凑齐长度所补的边，在长度计算中应该无碍
        # TODO：需验证一个或多个0，0边是否对长度无影响
        # if valid:
        #     return adj, log_probs, step+1
        return adj, log_probs, indexes


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        # embedder args
        self.d_input = 3
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder(self.d_input, self.d_model)
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        # self.gnn = TripleGNN(self.d_model, self.d_model, self.d_model)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs):
        pad = torch.zeros([inputs.shape[0], inputs.shape[1], 1], device=inputs.device)
        pad[:, 0, 0] = 1
        inputs = torch.cat([inputs, pad], -1)
        critic_encode = self.crit_encoder(self.crit_embedder(inputs), None)
        # critic_encode = self.gnn(adj, adj_in, adj_out, critic_encode)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions


if __name__ == '__main__':
    # batch_size, degree, device = 5, 5, 'cuda:6'
    # actor = Actor().to(device)
    # node = np.random.rand(batch_size, degree, 2)
    # node = torch.from_numpy(node).to(device).to(torch.float32)
    # adj, log_probs, indexes = actor(node)
    # print(indexes)
    # print('Hello')
    # print('Why')
    node_emb = torch.randn(3, 4, 5)
    visited = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
    graph_emb = GraphEmbed3(5, 10)
    a = graph_emb(node_emb, visited)
    print(a.shape)

