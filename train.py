import os
from collections import defaultdict
from os import path as osp
import time
from tqdm import tqdm

import numpy as np
import torch

from code.optim import *
import code.utils.utils as utils
from code.benchmarks.mtl_benchmark import get_benchmark_class


class MTLTrainer:
    def __init__(self, args):
        self.args = args
        self.benchmark = get_benchmark_class(args.benchmark)(args)
        self.balancer = get_method(args.balancer, compute_stats=args.compute_cnumber)

        self.model = self.benchmark.get_model(args)
        self.balancer.add_model_parameters(self.model)
        self.model = self.model.cuda()

        self.optimizer = self.benchmark.get_optim(self.model, args)
        self.scheduler = self.benchmark.get_scheduler(self.optimizer, args)

        if self.args.load_state:
            self.load_state(self.args.load_state)

        self.res_path = None
        self.train_loader_kwargs = {}
        self.valid_loader_kwargs = {}
        self.train_metrics = []
        self.valid_metrics = []

    def train_epoch(self):
        train_loader = torch.utils.data.DataLoader(self.benchmark.datasets['train'], **self.train_loader_kwargs)
        self.model.train()
        loss_total, task_losses = 0, defaultdict(float)

        pbar = tqdm(total=len(train_loader))
        fmtl_metrics = open(osp.join(self.res_path, 'mtl_metrics.txt'), 'a')

        for i, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            self.balancer.step_with_model(
                data=data[0].cuda(),
                targets={task_name: data[i+1].cuda() for i, task_name in enumerate(self.benchmark.task_names)},
                model=self.model,
                criteria=self.benchmark.task_criteria
            )
            self.optimizer.step()

            losses = self.balancer.losses
            if hasattr(self.balancer, 'info') and self.balancer.info is not None:
                fmtl_metrics.write(utils.strfy(self.balancer.info) + "\n")
                fmtl_metrics.flush()

            loss_total += sum(losses.values())
            for task_id in losses:
                task_losses[task_id] += losses[task_id]

            post = {"loss": sum(losses.values())}
            post.update(**losses)
            pbar.set_postfix(post)
            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        avg_total_loss = loss_total / len(train_loader)
        for task_id in task_losses:
            task_losses[task_id] /= len(train_loader)

        return avg_total_loss, task_losses

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.benchmark.datasets['valid'], **self.valid_loader_kwargs)
        loss_total = 0.0

        for data in test_loader:
            losses, _ = self.balancer.compute_losses(
                data=data[0].cuda(),
                targets={task_name: data[i+1].cuda() for i, task_name in enumerate(self.benchmark.task_names)},
                model=self.model,
                criteria=self.benchmark.task_criteria
            )
            loss_total += sum(losses.values())

        avg_val_loss = loss_total / len(test_loader)
        metrics = self.benchmark.evaluate(self.model, test_loader)
        return avg_val_loss, metrics

    def save_state(self, path):
        model_state = self.model.state_dict()
        torch.save({
            "state_dict": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        
    def load_state(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        # self.scheduler.load_state_dict(state['scheduler'])

    def run_experiment(self):
        utils.fix_seed(42 + self.args.round)
        train_kwargs = {
            "batch_size": self.args.train_batch,
            "drop_last": True,
            "shuffle": True,
        }
        test_kwargs = {"batch_size": self.args.test_batch, "shuffle": False}
        cuda_kwargs = {"num_workers": 8, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        self.train_loader_kwargs = train_kwargs
        self.valid_loader_kwargs = test_kwargs

        if self.args.eval_only:
            self.valid_epoch()
            return

        res_path = osp.join(self.args.output_path, self.args.benchmark, self.args.balancer, str(self.args.round))
        self.res_path = res_path
        if not osp.exists(res_path):
            os.makedirs(res_path)
        metrics_file_path = os.path.join(res_path, 'mtl_metrics.txt')
        if os.path.isfile(metrics_file_path):
            os.remove(metrics_file_path)

        best_val_loss = np.inf
        for epoch in range(self.args.epochs):
            print(f"Round: {args.round}; epoch: {epoch}")

            avg_train_loss, avg_task_losses = self.train_epoch()
            self.train_metrics.append({'train_loss': avg_train_loss, 'task_losses': avg_task_losses})
            print(f"Epoch: {epoch}, ", f"avg_train_loss: {avg_train_loss}, ", end=' ')
            for task_id in avg_task_losses:
                print('loss_{}: {:.4f}'.format(task_id, avg_task_losses[task_id]), end=', ')
            print()

            avg_val_loss, metrics = self.valid_epoch()
            self.valid_metrics.append({'val_loss': avg_val_loss, 'metrics': metrics})
            print(f"Epoch: {epoch}, avg_valid_loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                print(f"Save the model state")
                self.save_state(osp.join(self.res_path, "best_test.pth"))

            self.scheduler.step()


if __name__ == "__main__":
    parser = utils.common_argparser()
    args, _ = parser.parse_known_args()

    benchmark_type = get_benchmark_class(args.benchmark)
    specific_parser = benchmark_type.get_arg_parser(parser)
    args = specific_parser.parse_args()

    trainer = MTLTrainer(args)
    trainer.run_experiment()
