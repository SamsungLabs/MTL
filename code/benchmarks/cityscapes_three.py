import torch
import torch.nn.functional as F
from . import mtl_benchmark

from code.data.augmentation.cityscapes import *
from code.data.datasets.cityscapes import CITYSCAPES
from code.evaluation.cityscapes import CityScapesEvaluator
from code.models.cityscapes import ResNet50Dilated, SegmentationDecoder


def l1_loss_depth(input, target, val=False):
    mask = target > 0
    if mask.data.sum() < 1:
        return None

    lss = F.l1_loss(input[mask], target[mask], reduction="mean")
    return lss


def l1_loss_instance(input, target, val=False):
    mask = target != 250
    if mask.data.sum() < 1:
        return None

    lss = F.l1_loss(input[mask], target[mask], reduction="mean")
    return lss


class PSPNetCityscapes(mtl_benchmark.MTLModel):
    def __init__(self, n_classes=19):
        super().__init__()
        self.encoder = ResNet50Dilated(True)
        self.decoders["IS"] = SegmentationDecoder(num_class=2, task_type="R")
        self.decoders["SS"] = SegmentationDecoder(num_class=n_classes, task_type="C")
        self.decoders["DE"] = SegmentationDecoder(num_class=1, task_type="R")
        self.last_shared_layer = self.encoder.layer4


class CityscapesScheduler:
    def __init__(self, optim, milestone, period, scale):
        self.optim = optim
        self.epoch = 0
        self.milestone = milestone
        self.period = period
        self.scale = scale

    def state_dict(self):
        return {}

    def step(self):
        self.epoch += 1
        if self.epoch >= self.milestone and self.epoch % self.period == 0:
            for param_group in self.optim.param_groups:
                param_group["lr"] *= self.scale


@mtl_benchmark.register("cityscapes_pspnet")
class CityscapesBenchmark(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=["SS", "IS", "DE"],
            task_criteria={
                "SS": torch.nn.NLLLoss(ignore_index=250),
                "IS": l1_loss_instance,
                "DE": l1_loss_depth
            }
        )
        augs = Compose([RandomRotate(10), RandomHorizontallyFlip()])

        dataset1 = CITYSCAPES(
            root=args.data_path,
            is_transform=True,
            split=["train"],
            augmentations=augs,
        )
        dataset2 = CITYSCAPES(
            root=args.data_path, is_transform=True, split=["val"]
        )
        self.datasets = {
            'train': dataset1,
            'valid': dataset2
        }

    def evaluate(self, model, loader):
        return CityScapesEvaluator.evaluate(model, loader, device=next(model.parameters()).device)

    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=8, test_batch=4, epochs=60, lr=1e-4)
        parser.add_argument('--lr-milestone', type=int, default=40, help="Use LR scheduler after N epochs")
        parser.add_argument('--lr-period', type=int, default=3, help='LR decay frequency in epochs')
        parser.add_argument('--lr-scaler', type=float, default=0.7, help='LR decay factor')
        return parser

    @staticmethod
    def get_model(args):
        return PSPNetCityscapes()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    @staticmethod
    def get_scheduler(optimizer, args):
        return CityscapesScheduler(optimizer, milestone=args.lr_milestone, period=args.lr_period, scale=args.lr_scaler)


