import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_imagenet_mean_std():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std


def test_7scenes_img_augmentations():
    mean, std = get_imagenet_mean_std()
    crop_size = 224
    augs = A.Compose(
        [
            A.SmallestMaxSize(max_size=256, always_apply=True),
            A.CenterCrop(crop_size, crop_size, always_apply=True),
            A.Normalize(mean, std),
            ToTensorV2(),
        ],
        p=1,
    )
    return augs


def train_7scenes_img_augmentations():
    mean, std = get_imagenet_mean_std()
    crop_size = 224
    augs = A.Compose(
        [
            A.SmallestMaxSize(max_size=256, always_apply=True),
            A.RandomCrop(crop_size, crop_size, always_apply=True),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.2, p=0.7
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=0, p=0.7
                    ),
                    A.CLAHE(p=0.5),
                    A.ChannelShuffle(p=0.1),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(p=0.5),
                    A.Blur(p=0.5),
                ],
                p=0.3,
            ),
            A.GaussNoise(p=0.5),
            A.Normalize(mean, std),
            ToTensorV2(),
        ],
        p=1,
    )
    return augs


def get_7scenes_img_augmentations():
    train_augs = train_7scenes_img_augmentations()
    test_augs = test_7scenes_img_augmentations()
    return train_augs, test_augs
