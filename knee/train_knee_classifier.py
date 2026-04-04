# FILE: train_knee_classifier.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils.dataset import KneeDataset
from utils.augmentations import get_train_transform, get_val_transform
from models.encoder import DenseNetEncoder
from models.classifier import KneeClassifier

# ===============================
# Reproducibility
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ===============================
# Config
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = os.path.join("..", "datasets", "knee_oa")
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
NUM_CLASSES = 3
PATIENCE = 8

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Training
# ===============================
def train():
    # Datasets
    train_dataset = KneeDataset(
        DATA_ROOT, split="train", transform=get_train_transform()
    )
    val_dataset = KneeDataset(
        DATA_ROOT, split="test", transform=get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ===============================
    # Model
    # ===============================
    encoder = DenseNetEncoder(pretrained=True)
    encoder.freeze_all()
    encoder.unfreeze_last_block()

    model = KneeClassifier(encoder, num_classes=NUM_CLASSES).to(DEVICE)

    # ===============================
    # Class-weighted loss
    # ===============================
    counter = Counter()
    for _, label in train_dataset:
        counter[label] += 1

    total = sum(counter.values())
    class_weights = [
        total / (NUM_CLASSES * counter[i]) for i in range(NUM_CLASSES)
    ]
    class_weights = torch.tensor(class_weights).float().to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ===============================
    # Optimizer & Scheduler
    # ===============================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ===============================
    # Training loop
    # ===============================
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        train_preds, train_labels = [], []
        train_loss = 0.0

        for x, y in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
        ):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = out.argmax(dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_f1 = f1_score(train_labels, train_preds, average="macro")

        # ---------- VALIDATION ----------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # ---------- EARLY STOPPING ----------
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "best_knee_model.pth"),
            )
            print("🔥 New best model saved (by Macro-F1).")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    print(f"Training complete. Best Val Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
