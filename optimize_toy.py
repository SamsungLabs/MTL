import argparse
import os
from collections import defaultdict
from os import path as osp
from tqdm import tqdm

import numpy as np
import torch

from code.optim import *
import code.utils.utils as utils
import code.utils.toy as toy_problem
import code.utils.toy_plot as toy_plot

# Adapted from Nash-MTL: https://github.com/AvivNavon/nash-mtl/blob/main/experiments/toy/trainer.py


def main(method_type, device='cpu', n_iter=35000, scale=0.5):
    F = toy_problem.Toy(scale=scale)
    all_traj = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([0.0, 0.0]),
        torch.Tensor([9.0, 9.0]),
        torch.Tensor([-7.5, -0.5]),
        torch.Tensor([9, -1.0]),
    ]

    for i, init in enumerate(inits):
        traj = []
        x = init.clone()
        x = x.to(device).requires_grad_(True)

        balancer = get_method(method_type)
        optimizer = torch.optim.Adam([dict(params=[x], lr=1e-3)])

        for _ in tqdm(range(n_iter)):
            traj.append(x.cpu().detach().numpy().copy())

            optimizer.zero_grad()
            f = F(x, False)

            balancer.step(
                losses={"0": f[0], "1": f[1]},
                shared_params=[x, ],
                task_specific_params=None
            )
            optimizer.step()

        all_traj[i] = dict(init=init.cpu().detach().numpy().copy(), traj=np.array(traj))

    return all_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Challenging synthetic MTL example with two tasks")
    parser.add_argument('--iters', type=int, default=35000, help='Number iterations')
    parser.add_argument('--scale', type=float, default=0.5, help='Task balance')
    parser.add_argument('--balancer', type=str, default='amtl', help='MTL optimization method')

    args = parser.parse_args()
    all_traj = main(args.balancer, n_iter=args.iters, scale=args.scale)

    os.makedirs('toy_outputs', exist_ok=True)
    torch.save(all_traj, f"toy_outputs/{args.balancer}_{args.scale}.pth")

    ax, fig = toy_plot.plot_2d_pareto(all_traj, args.scale)
    fig.savefig(f'toy_outputs/plot_{args.balancer}_{args.scale}.pdf')
