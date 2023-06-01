import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoders = torch.nn.ModuleDict()
        self.last_shared_layer = None

    def forward(self, img):
        hrepr = self.encoder(img)
        res = {}
        for task_id in self.decoders:
            res[task_id] = self.decoders[task_id](hrepr)
        return res


class MTLBenchmark:
    def __init__(self, task_names, task_criteria: dict):
        self.task_names = task_names
        self.task_criteria = task_criteria
        self.datasets = None
        self.evaluator = None
        assert len(self.task_names) == len(self.task_criteria)

    @staticmethod
    def get_arg_parser(base_parser):
        return base_parser

    @staticmethod
    def get_model(args):
        raise NotImplementedError()

    @staticmethod
    def get_optim(model, args):
        raise NotImplementedError()

    @staticmethod
    def get_scheduler(optimizer, args):
        raise NotImplementedError()


class MTLBenchmarkRegistry:
    registry = {}


def get_benchmark_class(benchmark_name):
    if benchmark_name not in MTLBenchmarkRegistry.registry:
        raise ValueError("Benchmark named '{}' is not defined, valid benchmarks are: {}".format(
            benchmark_name, ', '.join(MTLBenchmarkRegistry.registry.keys())))

    return MTLBenchmarkRegistry.registry[benchmark_name]


def register(name):
    def _register(cls):
        MTLBenchmarkRegistry.registry[name] = cls
        return cls
    return _register
