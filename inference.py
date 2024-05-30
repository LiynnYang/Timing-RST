# -*- coding: utf-8 -*-
# @Time        : 2023/5/8 11:19
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : inference.py
# @env         : PyCharm
# @Description :
import argparse
import time

import numpy as np
import torch
# TODO: Attention!
from models.model_sorted import Actor
from data.dataset import RandomRawdataInference
from torch.utils.data import DataLoader
from utils.myutil import eval_len_from_adj, eval_distance
import time
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--parameter', type=str, default='beCalled', help='parameter save directory')
parser.add_argument('--degree', type=int, default=40, help='degree of nets')
parser.add_argument('--batch_size', type=int, default=1000, help='test batch size')
parser.add_argument('--eval_size', type=int, default=10000, help='eval set size')
parser.add_argument('--transform', type=int, default=8, help='transform')
parser.add_argument('--device', type=str, default='cuda:6', help='transform')
parser.add_argument("--weight", default=0.0, type=float, help='weight of radius in cost function.')
args = parser.parse_args()
# TODO:1
base_dir = 'save/' + args.parameter + '/' + str(args.weight) + '/trst'
# if args.weight == 0.0:
#     ckp_dir = base_dir + str(args.degree) + 'b.pt'
# else:
#     ckp_dir = base_dir + '2_' + str(args.degree) + '_' + str(args.weight) + 'b.pt'
# ckp_dir = base_dir + str(args.degree) + '_' + str(args.weight) + 'b.pt'
ckp_dir = base_dir + '30_' + str(args.weight) + 'b.pt'
print(ckp_dir)

test_data = 'data/test_data/array_degree{}_num{}.npy'.format(args.degree, args.eval_size)
# test_data = 'algorithms/baseline/iccad15_nets/superblue1_degree/superblue1(normal).nets_degree{}_num{}.npy'.format(args.degree, args.eval_size)
# test_data = 'random1000_5.npy'
device = torch.device(args.device)
checkpoint = torch.load(ckp_dir)
actor = Actor()
actor.to(device)
actor.load_state_dict(checkpoint['actor_state_dict'])

eval_dataset = RandomRawdataInference(args.eval_size, args.degree, file_path=test_data)
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
# TODO:2
filename = 'record.txt'


# with open(filename, 'a') as f:
#     f.write("# maxLength wireLength cnt lambda avgTime")

def transform_inputs(inputs, t):
    # 0 <= t <= 7
    xs = inputs[:, :, 0]
    ys = inputs[:, :, 1]
    if t >= 4:
        temp = xs
        xs = ys
        ys = temp
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return torch.stack([xs, ys], -1)

def transform_all(inputs): # [8, batchsize, degree, 2]
    ans = []
    for i in range(8):
        ans.append(transform_inputs(inputs, i))
    return torch.stack(ans, 0)


inference_time = 0.0
if args.transform == 1:
    eval_lengths, eval_radius, eval_tradeoff, error_batch = [], [], [], []
    for eval_batch in eval_loader:
        arrs = eval_batch
        arrs = arrs.to(device)
        # opt_len = opt_len.to(device_id)

        # xy_min = torch.min(arrs, dim=1, keepdim=True)[0]
        # xy_max = torch.max(arrs, dim=1, keepdim=True)[0]
        # arrs_normalized = (arrs - xy_min) / (xy_max - xy_min)

        with torch.no_grad():
            start_time = time.time()
            new_adj, _, indexs = actor(arrs, deterministic=True)
            inference_time += time.time() - start_time
            lengths = eval_len_from_adj(arrs, args.degree, new_adj)
            radius = np.array(eval_distance(arrs, indexs, [0] * arrs.shape[0]))
        eval_lengths.append(lengths.mean())
        eval_radius.append(radius.mean())
        eval_tradeoff.append((1 - args.weight) * lengths.mean() + args.weight * radius.mean())
        # error_batch.append(round(((lengths / opt_len).mean() - 1).item() * 100, 3))
    # mean_length, mean_radius, mean_tradeoff, mean_error = sum(eval_lengths) / len(eval_lengths), \
    #     sum(eval_radius) / len(eval_radius), sum(eval_tradeoff) / len(eval_tradeoff), sum(error_batch) / len(error_batch)
    mean_length, mean_radius, mean_tradeoff = sum(eval_lengths) / len(eval_lengths), \
                                              sum(eval_radius) / len(eval_radius), sum(eval_tradeoff) / len(
        eval_tradeoff)
    print("mean_length:{}, mean_radius:{}, mean_tradeoff:{}, mean_error:{}".format(mean_length, mean_radius,
                                                                                   mean_tradeoff, 0))
    with open(filename, 'a') as f:
        f.write('\n' + str(mean_radius) + ' ' + str(mean_length) + ' ' + str(args.eval_size) + ' ' + str(
            args.weight) + ' ' + str(inference_time / args.eval_size))

# else:  # transform=8
    # eval_lengths, eval_radius, eval_tradeoff, error_batch = [], [], [], []
    inference_time = 0
    # base = torch.tensor([[[0.5, 0.5]]]).to(device)
    # f = lambda d: 1 / (1 + d ** 2)
    for eval_batch in eval_loader:
        arrs = eval_batch
        arrs = arrs.to(device)
        # opt_len = opt_len.to(device_id)
        lengths_best, radius_best, tradeoff_best, error_best = [1e9 for i in range(arrs.shape[0])], [1e9 for i in range(
            arrs.shape[0])], [1e9 for i in range(arrs.shape[0])], [1e9 for i in range(arrs.shape[0])]
        t_arrs = transform_all(arrs).reshape(-1, args.degree, 2)
        with torch.no_grad():
            start_time = time.time()
            new_adj, _, indexs = actor(t_arrs, deterministic=True)
            lengths = eval_len_from_adj(t_arrs, args.degree, new_adj)
            length = np.array(length).reshape(8, -1, args.degree, 2)
            inference_time += time.time() - start_time
            radius = np.array(eval_distance(t_arrs, indexs, [0] * t_arrs.shape[0]))
            radius = radius.reshape(8, -1, args.degree, 2)
            tradeoff = (1 - args.weight) * lengths + args.weight * radius
        best_index = np.argmin(tradeoff, 0)
        best_tradeoff = np.min(tradeoff, 0)
        best_lengths = lengths[best_index, np.arange(arrs.shape[0])]
        best_radius = best_radius[best_index, np.arange(arrs.shape[0])]
        # for t in range(8):  # 8次翻转
        #     t_arrs = transform_inputs(arrs, t)
        #     # xy_min = torch.min(t_arrs, dim=1, keepdim=True)[0]
        #     # xy_max = torch.max(t_arrs, dim=1, keepdim=True)[0]
        #     # arrs_normalized = (t_arrs - xy_min) / (xy_max - xy_min)
        #     #
        #     # offsets = arrs_normalized - base
        #     # distances = f(offsets)
        #     # new_points = offsets * distances + base

        #     with torch.no_grad():
        #         start_time = time.time()
        #         new_adj, _, indexs = actor(t_arrs, deterministic=True)
        #         lengths = eval_len_from_adj(t_arrs, args.degree, new_adj)
        #         inference_time += time.time() - start_time
        #         radius = np.array(eval_distance(t_arrs, indexs, [0] * t_arrs.shape[0]))
        #         tradeoff = (1 - args.weight) * lengths + args.weight * radius
        #     for i in range(len(tradeoff_best)):
        #         if tradeoff[i] < tradeoff_best[i]:
        #             tradeoff_best[i] = tradeoff[i]
        #             lengths_best[i] = lengths[i]
        #             radius_best[i] = radius[i]

        eval_lengths.append(np.array(best_lengths).mean())
        eval_radius.append(np.array(best_radius).mean())
        eval_tradeoff.append(np.array(best_tradeoff).mean())
        error_batch.append(np.array(error_best).mean())
    mean_length, mean_radius, mean_tradeoff, mean_error = sum(eval_lengths) / len(eval_lengths), \
                                                          sum(eval_radius) / len(eval_radius), sum(
        eval_tradeoff) / len(eval_tradeoff), sum(error_batch) / len(error_batch)
    print("mean_length:{}, mean_radius:{}, mean_tradeoff:{}, mean_error:{}".format(mean_length, mean_radius,
                                                                                   mean_tradeoff, mean_error))
    with open(filename, 'a') as f:
        f.write('\n' + str(mean_radius) + ' ' + str(mean_length) + ' ' + str(args.eval_size) + ' ' + str(
            args.weight) + ' ' + str(inference_time / args.eval_size))


print(inference_time)