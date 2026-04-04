# FILE: utils/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose([
        A.Resize(224, 224),

        # Geometry (safe for knees)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=7, p=0.5),

        # Contrast enhancement (VERY important for X-rays)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),

        # Illumination robustness
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.5
        ),

        # Noise robustness
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.3
        ),

        # Normalize for ImageNet pretrained CNNs
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
