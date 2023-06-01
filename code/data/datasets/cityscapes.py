import os
from code.data.augmentation.cityscapes import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    :param rootdir is the root directory
    :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class CITYSCAPES(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        split=["train"],
        is_transform=False,
        img_size=(512, 256),
        augmentations=None,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = os.path.expanduser(root)
        self.split = split
        self.split_text = "+".join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array([123.675, 116.28, 103.53])
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, "leftImg8bit", _split)
            self.annotations_base = os.path.join(self.root, "gtFine", _split)
            self.files[self.split_text] = recursive_glob(
                rootdir=self.images_base, suffix=".png"
            )
            self.depth_base = os.path.join(self.root, "disparity", _split)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.DEPTH_STD = 2729.0680031169923
        self.DEPTH_MEAN = 0.0  # np.load('depth_mean.npy')

        if len(self.files[self.split_text]) < 2:
            raise Exception(
                "No files for split=[%s] found in %s"
                % (self.split_text, self.images_base)
            )

        print(
            "Found %d %s images" % (len(self.files[self.split_text]), self.split_text)
        )

    def __len__(self):
        """__len__"""
        return len(self.files[self.split_text])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        instance_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
        )
        depth_path = os.path.join(
            self.depth_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "disparity.png",
        )
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        ins = Image.open(instance_path)
        depth = np.array(Image.open(depth_path), dtype=np.float32)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        ins = np.array(ins, dtype=np.int32)
        depth = np.array(depth, dtype=np.float32)

        # depth[depth!=0] = (depth[depth!=0] - self.DEPTH_MEAN[depth!=0]) / self.DEPTH_STD
        depth = depth / self.DEPTH_STD

        if self.augmentations is not None:
            img, lbl, ins, depth = self.augmentations(img, lbl, ins, depth)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        ins_y, ins_x = self.encode_instancemap(lbl, ins)
        # Zero-Mean, Std-Dev depth map

        if self.is_transform:
            img, lbl, ins_gt, depth = self.transform(img, lbl, ins_y, ins_x, depth)

        return img, lbl, ins_gt, depth

    def transform(self, img, lbl, ins_y, ins_x, depth):
        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        # lbl = lbl.astype(int)
        lbl = cv2.resize(
            lbl,
            (int(self.img_size[0] / 8), int(self.img_size[1] / 8)),
            interpolation=cv2.INTER_NEAREST,
        )  # TODO(ozan) /8 is quite hacky
        # lbl = lbl.astype(int)

        # ins_y = ins_y.astype(int)
        ins_y = cv2.resize(
            ins_y,
            (int(self.img_size[0] / 8), int(self.img_size[1] / 8)),
            interpolation=cv2.INTER_NEAREST,
        )

        # ins_x = ins_x.astype(int)
        ins_x = cv2.resize(
            ins_x,
            (int(self.img_size[0] / 8), int(self.img_size[1] / 8)),
            interpolation=cv2.INTER_NEAREST,
        )

        depth = cv2.resize(
            depth,
            (int(self.img_size[0] / 8), int(self.img_size[1] / 8)),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, axis=0)
        # if not np.all(classes == np.unique(lbl)):
        #    print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        ins = np.stack((ins_y, ins_x))
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        ins = torch.from_numpy(ins).float()
        depth = torch.from_numpy(depth).float()
        return img, lbl, ins, depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def encode_instancemap(self, mask, ins):
        ins[mask == self.ignore_index] = self.ignore_index
        for _no_instance in self.no_instances:
            ins[ins == _no_instance] = self.ignore_index
        ins[ins == 0] = self.ignore_index

        instance_ids = np.unique(ins)
        sh = ins.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing="ij")

        out_ymap, out_xmap = np.meshgrid(
            np.arange(sh[0]), np.arange(sh[1]), indexing="ij"
        )
        out_ymap = np.ones(ymap.shape) * self.ignore_index
        out_xmap = np.ones(xmap.shape) * self.ignore_index

        for instance_id in instance_ids:
            if instance_id == self.ignore_index:
                continue
            instance_indicator = ins == instance_id
            coordinate_y, coordinate_x = np.mean(ymap[instance_indicator]), np.mean(
                xmap[instance_indicator]
            )
            out_ymap[instance_indicator] = ymap[instance_indicator] - coordinate_y
            out_xmap[instance_indicator] = xmap[instance_indicator] - coordinate_x

        return out_ymap, out_xmap
