import numpy as np
import torch
import torch.nn as nn
import os
import math


class Embedder(nn.Module):
    def __init__(self, d_input, d_model):
        super(Embedder, self).__init__()
        self.conv1d = nn.Conv1d(d_input, d_model, 1)
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, inputs):
        embeddings = self.conv1d(inputs.permute(0, 2, 1))
        embeddings = self.batch_norm(embeddings).permute(0, 2, 1)
        return embeddings


class Pointer(nn.Module):
    def __init__(self, d_query, d_unit):
        super(Pointer, self).__init__()
        self.tanh = nn.Tanh()
        self.w_l = nn.Linear(d_query, d_unit, bias=False)
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))

    def forward(self, edge_emb, query, mask=None):
        """

        :param edge_emb: batch*edge*edge_dim
        :param query: batch*query*dim
        :param mask: batch*edge*edge
        :return: scores:batch*(degree*degree)
        """
        batch = edge_emb.shape[0]
        scores = torch.sum(self.v * self.tanh(edge_emb + self.w_l(query).unsqueeze(1)), -1)
        # scores = torch.sum(self.v * self.tanh(edge_emb), -1)
        scores = 10. * self.tanh(scores)
        with torch.no_grad():
            if mask is not None:
                mask = mask.reshape(batch, -1)
                scores[mask == 0] = float('-inf')
                # 如果mask全零，则选择0，0,避免报错
                t1 = torch.all(mask == 0, dim=1)
                t2 = torch.where(t1==True)[0]
                scores[t2, 0] = 0

                # scores[mask] = float('-inf')
        # scores = F.softmax(scores, dim=-1)
        if True in torch.isnan(scores):
            print("存在Nan！！！！！！！！！！！！！！！！")
        return scores, t2


class Glimpse(nn.Module):
    def __init__(self, d_model, d_unit):
        super(Glimpse, self).__init__()
        self.tanh = nn.Tanh()
        self.conv1d = nn.Conv1d(d_model, d_unit, 1)
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs, mask=None):
        encoded = self.conv1d(encs.permute(0, 2, 1)).permute(0, 2, 1)
        scores = torch.sum(self.v * self.tanh(encoded), -1)
        if mask is not None:  # mask为1代表是pad进来的
            scores[mask] = float('-inf')
        attention = self.softmax(scores)
        glimpse = attention.unsqueeze(-1) * encs
        glimpse = torch.sum(glimpse, 1)
        return glimpse


if __name__ == '__main__':
    glimpse = Glimpse(128, 256)
    encs = torch.FloatTensor(10, 20, 128)
    a = glimpse(encs)