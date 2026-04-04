# FILE: utils/dataset.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class KneeDataset(Dataset):
    """
    Knee Osteoarthritis Dataset
    - Original KL grades: 0–4
    - Mapped to 3 classes:
        0 → No / Doubtful OA (KL 0–1)
        1 → Mild–Moderate OA (KL 2–3)
        2 → Severe OA (KL 4)
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels_5class = []

        split_dir = os.path.join(root_dir, split)

        for grade in ["0", "1", "2", "3", "4"]:
            grade_dir = os.path.join(split_dir, grade)
            if not os.path.isdir(grade_dir):
                continue

            for img_name in os.listdir(grade_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(grade_dir, img_name))
                    self.labels_5class.append(int(grade))

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def roi_crop(img):
        """
        Conservative knee ROI crop.
        Works safely across different resolutions.
        """
        h, w = img.shape[:2]
        y1, y2 = int(0.35 * h), int(0.75 * h)
        x1, x2 = int(0.20 * w), int(0.80 * w)
        return img[y1:y2, x1:x2]

    @staticmethod
    def map_to_3class(kl_grade):
        if kl_grade <= 1:
            return 0
        elif kl_grade <= 3:
            return 1
        else:
            return 2

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # Handle grayscale images safely
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.roi_crop(img)

        label_5 = self.labels_5class[idx]
        label_3 = self.map_to_3class(label_5)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label_3
