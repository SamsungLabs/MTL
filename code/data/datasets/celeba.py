import glob
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data


class CELEBA(data.Dataset):
    class_names = [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(32, 32),
        augmentations=None,
    ):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 40
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])

        self.files = {}
        self.labels = {}

        label_map = {}

        with open(self.root + "/Anno/list_attr_celeba.txt", "r") as file:
            labels = file.read().split("\n")[2:-1]

        for label_line in labels:
            f_name = label_line.split(" ")[0]
            label_txt = list(
                map(lambda x: int(x), re.sub("-1", "0", label_line).split()[1:])
            )
            label_map[f_name] = label_txt

        all_files = glob.glob(self.root + "/Img/img_align_celeba/*.jpg")

        with open(self.root + "/Eval/list_eval_partition.txt", "r") as file:
            fl = file.read().split("\n")
            fl.pop()
            if "train" in self.split:
                selected_files = list(filter(lambda x: x.split(" ")[1] == "0", fl))
            elif "val" in self.split:
                selected_files = list(filter(lambda x: x.split(" ")[1] == "1", fl))
            elif "test" in self.split:
                selected_files = list(filter(lambda x: x.split(" ")[1] == "2", fl))
            selected_file_names = [x.split(" ")[0] for x in selected_files]

        base_path = "/".join(all_files[0].split("/")[:-1])
        intersect = map(lambda x: x.split("/")[-1], all_files)
        intersect = set(intersect).intersection(set(selected_file_names))

        self.files[self.split] = list(
            map(lambda x: "/".join([base_path, x]), intersect)
        )
        self.labels[self.split] = list(map(lambda x: label_map[x], intersect))

        if len(self.files[self.split]) < 2:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.root)
            )

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        img = np.array(Image.open(img_path))

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        if self.is_transform:
            img = self.transform_img(img)

        return img, torch.as_tensor(label)

    def transform_img(self, img):

        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean

        img = cv2.resize(img, dsize=(self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img
