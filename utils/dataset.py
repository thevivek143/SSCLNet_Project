import os
from PIL import Image
from torch.utils.data import Dataset
import glob

class BrainMRIDataset(Dataset):
    """
    Dataset for Brain MRI Classification.
    Expected structure: datasets/brain_mri/{class_name}/*.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            # Support multiple extensions
            image_paths = glob.glob(os.path.join(cls_folder, "*.*"))
            for img_path in image_paths:
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class KneeOADataset(Dataset):
    """
    Dataset for Knee Osteoarthritis Classification.
    Expected structure: datasets/knee_oa/{split}/{grade}/*.png
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        # Reviewing user request: datasets/knee_oa/train/0,1,2,3,4
        self.classes = ['0', '1', '2', '3', '4'] # Fixed classes based on KL grades
        self.images = []
        self.labels = []

        # Check if split directory exists
        if not os.path.exists(self.split_dir):
            print(f"Warning: Directory {self.split_dir} does not exist.")
            return

        for cls_name in self.classes:
            cls_folder = os.path.join(self.split_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
                
            image_paths = glob.glob(os.path.join(cls_folder, "*.*"))
            for img_path in image_paths:
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(img_path)
                    self.labels.append(int(cls_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class PretrainingDataset(Dataset):
    """
    Dataset wrapper for Self-Supervised Learning.
    Returns two augmented views of the same image.
    Labels are ignored.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We access the underlying image path directly if possible or load it
        # Since self.dataset is likely BrainMRIDataset, let's look at its implementation.
        # It's cleaner to reuse the internal logic or just call basic getitem and re-open?
        # Actually, BrainMRIDataset returns (image, label).
        # We need the raw PIL image to apply two different stochastic transforms.
        
        # Accessing private attribute strictly speaking, but for this project it's fine.
        img_path = self.dataset.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform twice to get two views
        x1 = self.transform(image)
        x2 = self.transform(image)
        
        return x1, x2
