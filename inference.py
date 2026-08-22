# -*- coding: utf-8 -*-
# @Time        : 2023/5/8 11:19
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : inference.py
# @Description : Moderate-degree Actor inference, plus polar quadtree divide-and-merge
#                for nets larger than 32 pins (paper Section III).
import argparse
import os
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--parameter', type=str, default='beCalled', help='parameter save directory')
parser.add_argument('--degree', type=int, default=30, help='degree of nets')
parser.add_argument('--batch_size', type=int, default=1024, help='test batch size')
parser.add_argument('--eval_size', type=int, default=10000, help='eval set size')
parser.add_argument('--transform', type=int, default=8, help='transform')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument("--weight", default=0.0, type=float, help='weight of radius in cost function.')
parser.add_argument('--capacity', type=int, default=30, help='polar block capacity B')
parser.add_argument(
    '--fallback_mst',
    action='store_true',
    help='use rectilinear MST as the local solver instead of the Actor',
)
args = parser.parse_args()

base_dir = 'save/' + args.parameter + '/' + str(args.weight) + '/trst'
ckp_dir = base_dir + '30_' + str(args.weight) + 'b.pt'
print(ckp_dir)


def transform_inputs(inputs, t):
    import torch
    xs = inputs[:, :, 0]
    ys = inputs[:, :, 1]
    if t >= 4:
        xs, ys = ys, xs
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return torch.stack([xs, ys], -1)


def transform_all(inputs):
    import torch
    return torch.stack([transform_inputs(inputs, i) for i in range(8)], 0)


def infer_moderate():
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import RandomRawdataInference
    from models.model_sorted import Actor
    from utils.myutil import eval_distance, eval_len_from_adj

    device = torch.device(args.device)

    test_data = 'data/test_data/array_degree{}_num{}.npy'.format(args.degree, args.eval_size)
    checkpoint = torch.load(ckp_dir, map_location=device)
    actor = Actor()
    actor.to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    eval_dataset = RandomRawdataInference(args.eval_size, args.degree, file_path=test_data)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    filename = 'record.txt'
    inference_time = 0.0
    if args.transform == 1:
        eval_lengths, eval_radius, eval_tradeoff = [], [], []
        for eval_batch in eval_loader:
            arrs = eval_batch.to(device)
            with torch.no_grad():
                start_time = time.time()
                new_adj, _, indexs = actor(arrs, deterministic=True)
                inference_time += time.time() - start_time
            lengths = eval_len_from_adj(arrs, args.degree, new_adj)
            radius = np.array(eval_distance(arrs, indexs, [0] * arrs.shape[0]))
            eval_lengths.append(lengths.mean())
            eval_radius.append(radius.mean())
            eval_tradeoff.append((1 - args.weight) * lengths.mean() + args.weight * radius.mean())
        mean_length = sum(eval_lengths) / len(eval_lengths)
        mean_radius = sum(eval_radius) / len(eval_radius)
        mean_tradeoff = sum(eval_tradeoff) / len(eval_tradeoff)
        print("mean_length:{}, mean_radius:{}, mean_tradeoff:{}, mean_error:{}".format(
            mean_length, mean_radius, mean_tradeoff, 0))
        with open(filename, 'a') as f:
            f.write('\n{} {} {} {} {}'.format(
                mean_radius, mean_length, args.eval_size, args.weight,
                inference_time / args.eval_size))
    else:
        eval_lengths, eval_radius, eval_tradeoff = [], [], []
        for eval_batch in tqdm(eval_loader):
            arrs = eval_batch.to(device)
            t_arrs = transform_all(arrs).reshape(-1, args.degree, 2)
            with torch.no_grad():
                start_time = time.time()
                new_adj, _, indexs = actor(t_arrs, deterministic=True)
                lengths = np.array(eval_len_from_adj(t_arrs, args.degree, new_adj)).reshape(8, -1)
                inference_time += time.time() - start_time
            radius = np.array(eval_distance(t_arrs, indexs, [0] * t_arrs.shape[0])).reshape(8, -1)
            tradeoff = (1 - args.weight) * lengths + args.weight * radius
            best_index = np.argmin(tradeoff, 0)
            best_tradeoff = np.min(tradeoff, 0)
            best_lengths = lengths[best_index, np.arange(arrs.shape[0])]
            best_radius = radius[best_index, np.arange(arrs.shape[0])]
            eval_lengths.append(np.array(best_lengths).mean())
            eval_radius.append(np.array(best_radius).mean())
            eval_tradeoff.append(np.array(best_tradeoff).mean())
        mean_length = sum(eval_lengths) / len(eval_lengths)
        mean_radius = sum(eval_radius) / len(eval_radius)
        mean_tradeoff = sum(eval_tradeoff) / len(eval_tradeoff)
        print("mean_length:{}, mean_radius:{}, mean_tradeoff:{},".format(
            mean_length, mean_radius, mean_tradeoff))
        with open(filename, 'a') as f:
            f.write('\n{} {} {} {} {}'.format(
                mean_radius, mean_length, args.eval_size, args.weight,
                inference_time / args.eval_size))
    print('inference_time:', inference_time)


def _tree_metrics(points, tree):
    nodes = tree["nodes"]
    n = len(points)
    adj = [[] for _ in range(len(nodes))]
    wl = 0.0
    for a, b in tree["edges"]:
        adj[a].append(b)
        adj[b].append(a)
        pa, pb = nodes[a], nodes[b]
        wl += abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
    dist = [-1.0] * len(nodes)
    dist[0] = 0.0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if dist[v] >= 0:
                continue
            pu, pv = nodes[u], nodes[v]
            dist[v] = dist[u] + abs(pu[0] - pv[0]) + abs(pu[1] - pv[1])
            stack.append(v)
    radius = max((dist[i] for i in range(n) if dist[i] >= 0), default=0.0)
    connected = all(dist[i] >= 0 for i in range(n))
    return wl, radius, connected


def infer_large():
    from utils.divide_merge import divide_and_merge, rectilinear_mst

    rng = np.random.RandomState(0)
    cases = rng.rand(args.eval_size, args.degree, 2)
    if args.fallback_mst:
        solver = rectilinear_mst
    else:
        from utils.nn_solver import TimingRSTSolver
        if not os.path.exists(ckp_dir):
            raise FileNotFoundError(
                "checkpoint not found: {}. Use --fallback_mst to run without the Actor.".format(ckp_dir)
            )
        solver = TimingRSTSolver(
            ckp_dir, degree=min(args.capacity, 30), device=args.device, transform=args.transform
        )

    lengths, radii = [], []
    t0 = time.time()
    for i in tqdm(range(args.eval_size)):
        pts = [tuple(p) for p in cases[i]]
        tree = divide_and_merge(pts, solver, capacity=args.capacity)
        wl, rd, ok = _tree_metrics(pts, tree)
        if not ok:
            raise RuntimeError("divide-and-merge produced a disconnected tree on sample {}".format(i))
        lengths.append(wl)
        radii.append(rd)
    elapsed = time.time() - t0
    mean_length = float(np.mean(lengths))
    mean_radius = float(np.mean(radii))
    mean_tradeoff = (1 - args.weight) * mean_length + args.weight * mean_radius
    print("mean_length:{}, mean_radius:{}, mean_tradeoff:{},".format(
        mean_length, mean_radius, mean_tradeoff))
    print("inference_time:", elapsed)


if __name__ == '__main__':
    if args.degree > 32:
        infer_large()
    else:
        infer_moderate()
