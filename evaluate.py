import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from models.encoder import get_encoder
from models.classifier import get_brain_classifier, get_knee_classifier
from utils.dataset import BrainMRIDataset, KneeOADataset
from utils.augmentations import get_val_transform
from utils.metrics import calculate_metrics
from torch.utils.data import DataLoader, random_split
import os

def evaluate_models():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Evaluate Brain Model
    if os.path.exists("models/brain_model.pth"):
        print("\nEvaluating Brain MRI Model...")
        # Recreate dataset split logic (approximate for demo or use provided test set if it existed)
        root_dir = "datasets/brain_mri"
        ds_raw = BrainMRIDataset(root_dir, transform=get_val_transform())
        # Use simple split for demo consistency
        train_size = int(0.8 * len(ds_raw))
        val_size = len(ds_raw) - train_size
        _, val_subset = random_split(ds_raw, [train_size, val_size])
        
        # We need a transform wrapper like in train script if we want to be precise, 
        # but here we initialized with val_transform so it's fine.
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
        
        encoder = get_encoder()
        model = get_brain_classifier(encoder, num_classes=4).to(DEVICE)
        model.load_state_dict(torch.load("models/brain_model.pth", map_location=DEVICE, weights_only=True))
        model.eval()
        
        all_preds = []
        all_labels = []
        from tqdm import tqdm
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images = images.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print("Brain MRI Metrics:", calculate_metrics(all_labels, all_preds))
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=ds_raw.classes, 
                    yticklabels=ds_raw.classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Brain MRI Confusion Matrix')
        save_path = "brain_confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    # Evaluate Knee Model
    if os.path.exists("models/knee_model.pth"):
        print("\nEvaluating Knee OA Model...")
        root_dir = "datasets/knee_oa"
        val_dataset = KneeOADataset(root_dir, split='test', transform=get_val_transform())
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        encoder = get_encoder()
        model = get_knee_classifier(encoder, num_classes=5).to(DEVICE)
        model.load_state_dict(torch.load("models/knee_model.pth", map_location=DEVICE, weights_only=True))
        model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print("Knee OA Metrics:", calculate_metrics(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_models()
