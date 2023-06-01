import torch
import torch.nn.functional as F
from . import mtl_benchmark


from code.models.nyu2 import (
    DepthDecoder,
    NormalDecoder,
    ResNet50Dilated,
    SemanticDecoder,
)
from code.models.segnet_mtan import MTANEncoder, MTANDepthDecoder, MTANNormalDecoder, MTANSemanticDecoder
from code.data.datasets import NYUv2
from code.evaluation.nyu2 import NYUv2Evaluator


def semantic_loss(x_pred, target):
    # semantic loss: depth-wise cross entropy
    loss = F.nll_loss(x_pred, target.long(), ignore_index=-1)
    return loss


def depth_loss(x_pred, target):
    device = x_pred.device
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(device)
    # depth loss: l1 norm
    loss = torch.sum(torch.abs(x_pred - target) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


def normals_loss(x_pred, target):
    device = x_pred.device
    binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(device)
    # normal loss: dot product
    loss = 1 - torch.sum((x_pred * target) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


class PSPNetNYUModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50Dilated(True)
        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = self.encoder.layer4


class MTANNYUModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = MTANEncoder()
        self.decoders["SS"] = MTANSemanticDecoder(class_nb=13)
        self.decoders["NE"] = MTANNormalDecoder()
        self.decoders["DE"] = MTANDepthDecoder()
        self.last_shared_layer = None


@mtl_benchmark.register("nyuv2_pspnet")
class NYUBenchmark(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=["SS", "DE", "NE"],
            task_criteria={
                "SS": semantic_loss,
                "DE": depth_loss,
                "NE": normals_loss
            }
        )
        self.datasets = {
            'train': NYUv2(root=args.data_path, train=True, augmentation=True),
            'valid': NYUv2(root=args.data_path, train=False)
        }

    def evaluate(self, model, loader):
        return NYUv2Evaluator.evaluate(model, loader, device=next(model.parameters()).device)

    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=2, test_batch=16, epochs=40, lr=1e-4)
        parser.add_argument('--lr-decay-steps', type=int, default=20, help="Decrease LR every N epochs")
        parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='LR decay factor')
        parser.add_argument('--data-path', type=str, default=None, help='Path to NYUv2 dataset')
        return parser

    @staticmethod
    def get_model(args):
        return PSPNetNYUModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_factor)


@mtl_benchmark.register("nyuv2_mtan")
class NYUBenchmarkMTAN(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=["SS", "DE", "NE"],
            task_criteria={
                "SS": semantic_loss,
                "DE": depth_loss,
                "NE": normals_loss
            }
        )
        self.datasets = {
            'train': NYUv2(root=args.data_path, train=True, augmentation=True),
            'valid': NYUv2(root=args.data_path, train=False)
        }

    def evaluate(self, model, loader):
        return NYUv2Evaluator.evaluate(model, loader, device=next(model.parameters()).device)

    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=2, test_batch=16, epochs=200, lr=1e-4)
        parser.add_argument('--lr-decay-steps', type=int, default=100, help="Decrease LR every N epochs")
        parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='LR decay factor')
        parser.add_argument('--data-path', type=str, default=None, help='Path to NYUv2 dataset')
        return parser

    @staticmethod
    def get_model(args):
        return MTANNYUModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_factor)
