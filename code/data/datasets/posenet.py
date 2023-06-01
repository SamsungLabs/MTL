from code.data.augmentation.posenet import get_7scenes_img_augmentations
from os import path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def cal_quat_angle_error(label, pred):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))  # Here we have abs()
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error


class SevenScenesDatasetFactory(object):
    @staticmethod
    def _is_scene_valid(scene_name: str):
        return scene_name in {
            "chess",
            "fire",
            "heads",
            "office",
            "pumpkin",
            "redkitchen",
            "stairs",
        }

    @staticmethod
    def create_splits(data_path, scene_name):
        train_augs, test_augs = get_7scenes_img_augmentations()
        if not SevenScenesDatasetFactory._is_scene_valid(scene_name):
            raise ValueError(f"Not valid scene_name.")
        if scene_name == "chess":
            train_split = ChessDataset(data_path, "train", train_augs)
            test_split = ChessDataset(data_path, "val", test_augs)
        elif scene_name == "fire":
            train_split = FireDataset(data_path, "train", train_augs)
            test_split = FireDataset(data_path, "val", test_augs)
        elif scene_name == "heads":
            train_split = HeadsDataset(data_path, "train", train_augs)
            test_split = HeadsDataset(data_path, "val", test_augs)
        elif scene_name == "office":
            train_split = OfficeDataset(data_path, "train", train_augs)
            test_split = OfficeDataset(data_path, "val", test_augs)
        elif scene_name == "pumpkin":
            train_split = PumpkinDataset(data_path, "train", train_augs)
            test_split = PumpkinDataset(data_path, "val", test_augs)
        elif scene_name == "redkitchen":
            train_split = RedkitchenDataset(data_path, "train", train_augs)
            test_split = RedkitchenDataset(data_path, "val", test_augs)
        elif scene_name == "stairs":
            train_split = StairsDataset(data_path, "train", train_augs)
            test_split = StairsDataset(data_path, "val", test_augs)

        return train_split, test_split


class SevenScenesBase(object):
    def __init__(self, data_path: str, split: str):
        self.data_path = data_path

        if SevenScenesBase.is_split_valid(split):
            self.split = split
        else:
            raise ValueError(
                f"Check the value. The split should be either train or val."
            )

    def get_gt_data(self, scene_name):
        gt_fname = osp.join(
            self.data_path, scene_name, f"{scene_name}_{self.split}.txt"
        )
        fnames, t_gt, q_gt = SevenScenesBase.read_txt_poses(gt_fname)

        # create fullpath
        fnames = [osp.join(self.data_path, scene_name, fname[1:]) for fname in fnames]
        return fnames, t_gt, q_gt

    @staticmethod
    def read_txt_poses(fname: str):
        line_to_skip = 3
        fnames, t_arr, q_arr = [], [], []
        with open(fname) as f:
            for _ in range(line_to_skip):
                next(f)
            for line in f:
                chunks = line.rstrip().split(" ")
                fnames.append(chunks[0])
                t_arr.append(
                    torch.FloatTensor(
                        [float(chunks[1]), float(chunks[2]), float(chunks[3])]
                    )
                )
                q_arr.append(
                    torch.FloatTensor(
                        [
                            float(chunks[4]),
                            float(chunks[5]),
                            float(chunks[6]),
                            float(chunks[7]),
                        ]
                    )
                )
        return fnames, t_arr, q_arr

    @staticmethod
    def is_split_valid(split_name: str):
        return split_name in ["train", "val"]


class SceneBase(Dataset, SevenScenesBase):
    def __init__(self, data_path: str, split: str, scene_name: str, transforms=None):
        super(SceneBase, self).__init__(data_path, split)
        self.transforms = transforms
        self.fnames, self.t_gt, self.q_gt = self.get_gt_data(scene_name)

    def __getitem__(self, item):
        img = Image.open(self.fnames[item]).convert("RGB")
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]

        if self.transforms:
            img = self.transforms(image=np.asarray(img))["image"]

        return {"img": img, "t_gt": t_gt, "q_gt": q_gt}

    def __len__(self):
        return len(self.fnames)


class FireDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(FireDataset, self).__init__(data_path, split, "fire", transforms)


class ChessDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(ChessDataset, self).__init__(data_path, split, "chess", transforms)


class HeadsDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(HeadsDataset, self).__init__(data_path, split, "heads", transforms)


class OfficeDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(OfficeDataset, self).__init__(data_path, split, "office", transforms)


class PumpkinDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(PumpkinDataset, self).__init__(data_path, split, "pumpkin", transforms)


class RedkitchenDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(RedkitchenDataset, self).__init__(
            data_path, split, "redkitchen", transforms
        )


class StairsDataset(SceneBase):
    def __init__(self, data_path: str, split: str, transforms=None):
        super(StairsDataset, self).__init__(data_path, split, "stairs", transforms)
