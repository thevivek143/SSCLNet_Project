# FILE: evaluate_knee_classifier.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from utils.dataset import KneeDataset
from utils.augmentations import get_val_transform
from models.encoder import DenseNetEncoder
from models.classifier import KneeClassifier

# ===============================
# Config
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = os.path.join("..", "datasets", "knee_oa")
MODEL_PATH = os.path.join("models", "best_knee_model.pth")
BATCH_SIZE = 16
NUM_CLASSES = 3

CLASS_NAMES = [
    "No / Doubtful OA",
    "Mild–Moderate OA",
    "Severe OA",
]

# ===============================
# Evaluation
# ===============================
def evaluate():
    # Dataset
    val_dataset = KneeDataset(
        DATA_ROOT, split="test", transform=get_val_transform()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Evaluation samples: {len(val_dataset)}")

    # Model
    encoder = DenseNetEncoder(pretrained=False)
    model = KneeClassifier(encoder, num_classes=NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ===============================
    # Metrics
    # ===============================
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n📊 Evaluation Metrics")
    print(f"Accuracy      : {acc*100:.2f}%")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    print("\n📋 Classification Report")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    # ===============================
    # Confusion Matrix
    # ===============================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – Knee OA Classification")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
