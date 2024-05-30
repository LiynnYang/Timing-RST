# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/28 19:50
# @Author      : gwsun
# @Project     : RSMT
# @File        : dataset.py
# @env         : PyCharm
# @Description :
import os

from torch.utils.data import Dataset, DataLoader
from data.get_merge_data import get_data
from utils.myutil import get_length_batch
# from ../coreset/REST-main-111/coreset import get_coreset
import torch
import numpy as np


class RandomRawdata(Dataset):
    """
    最简单的数据集，只包含随机生成的数据坐标
    """
    def __init__(self, num, degree, file_path=None, use_coreset=True):  # num = batched * batch_size
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        if not use_coreset or file_path is None or not os.path.exists(file_path):
            print("data random generated.")
            cases = np.random.rand(num, degree, 2)
            unsampled_cases = np.round(cases, 8)
            self.arr = torch.from_numpy(unsampled_cases).to(torch.float32)

        else:
            cases = np.load(file_path)
            cases = np.round(cases, 8)
            assert num == len(cases), 'Dataset size:{} is not match parameter num:{}!'.format(len(cases), num)
            assert degree == len(cases[0]), 'Degree is not match!'
            print('coreset read successfully, the size is {}'.format(num))
            self.arr = torch.from_numpy(cases).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index]

    def __len__(self):
        return len(self.arr)


class RandomRawdataEval(Dataset):
    """
    最简单的数据集，只包含随机生成的数据坐标, 并且带有最优解长度
    """
    def __init__(self, num, degree, file_path=None):  # num = batched * batch_size
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        if file_path is None or not os.path.exists(file_path):
            cases = np.random.rand(num, degree, 2)
            unsampled_cases = np.round(cases, 8)
            self.arr = torch.from_numpy(unsampled_cases).to(torch.float32)
            self.gst_lengths = get_length_batch(unsampled_cases)
            np.save('data/test_data/array_degree{}_num{}.npy'.format(degree, num), unsampled_cases)
            np.save('data/test_data/length_degree{}_num{}.npy'.format(degree, num), self.gst_lengths)
        else:
            unsampled_cases = np.load('data/test_data/array_degree{}_num{}.npy'.format(degree, num))
            self.arr = torch.from_numpy(unsampled_cases).to(torch.float32)
            self.gst_lengths = np.load('data/test_data/length_degree{}_num{}.npy'.format(degree, num))

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.gst_lengths[index]

    def __len__(self):
        return len(self.arr)


class RandomRawdataInference(Dataset):
    """
    最简单的数据集，只包含随机生成的数据坐标, 并且带有最优解长度
    """
    def __init__(self, num, degree, file_path=None):  # num = batched * batch_size
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        if file_path is None or not os.path.exists(file_path):
            print('no test file, must!')
            exit(0)
        else:
            unsampled_cases = np.load(file_path)
            self.arr = torch.from_numpy(unsampled_cases).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index]

    def __len__(self):
        return len(self.arr)
class RandomDataset(Dataset):
    """
    此类获取的数据集不含有原长度，只适用于训练
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./data'):
        arr, adj, adj_in, adj_out, mask, opt_len = get_data(num, degree, m_block, block_max, file=file)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index]

    def __len__(self):
        return len(self.arr)


class RandomDataset2(Dataset):
    """
    此类获取数据集含有原长度以及两块的大小，用于测试
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./datamini'):
        arr, adj, adj_in, adj_out, mask, opt_len, ori_len, ori_size, link_b = get_data(num, degree, m_block, block_max, file=file, mode=2)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)
        self.ori_len = torch.from_numpy(ori_len).to(torch.float32)
        self.ori_size = torch.from_numpy(ori_size).to(torch.int64)
        self.link_b = torch.from_numpy(link_b).to(torch.int64)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index], self.ori_len[index], self.ori_size[index], self.link_b[index]

    def __len__(self):
        return len(self.arr)


class RandomDataset3(Dataset):
    """
    此类只含有原长度，不含有大小，适用于测试
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./datamini'):
        arr, adj, adj_in, adj_out, mask, opt_len, ori_len = get_data(num, degree, m_block, block_max, file=file, mode=2)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)
        self.ori_len = torch.from_numpy(ori_len).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index], self.ori_len[index]

    def __len__(self):
        return len(self.arr)


if __name__ == '__main__':
    degree = 40
    dataset = RandomDataset(100000, 40)
    dataloador = DataLoader(dataset, batch_size=256)
    for sample in dataloador:
        print(sample)
        break
