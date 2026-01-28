import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.encoder import get_encoder
from models.classifier import get_brain_classifier
from utils.dataset import BrainMRIDataset
from utils.augmentations import get_train_transform, get_val_transform
from utils.metrics import calculate_metrics
import os

def train_brain():
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 1
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Dataset
    root_dir = "datasets/brain_mri"
    full_dataset = BrainMRIDataset(root_dir, transform=None) # We apply transforms later or wrapper
    # Actually, BrainMRIDataset takes transform in init.
    # To split, we'd ideally split indices, then apply different transforms.
    # For simplicity in this demo, let's use the train_transform for everything in training loop, 
    # or recreate datasets with subset.
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # We need to hack the transform because Subset doesn't allow changing transform easily
    # A cleaner way is to define a Wrapper or just pass the transform to the loader logic 
    # if the dataset supported on-the-fly transform (it doesn't, it supports init transform).
    # Let's override the transform attribute of the underlying dataset? No, that affects both.
    # Re-instantiating datasets is better if we had file lists.
    # Let's rely on `utils/dataset.py` BrainMRIDataset. It pulls all files.
    # Best way: Split the file paths manually.
    
    # Quick fix: Just use train transform for all for simplicity, or
    # Re-impl Dataset to accept list of files. 
    # BUT, to keep it simple and given constraints:
    # Just assign the transform to the dataset instance now.
    full_dataset.transform = get_train_transform()
    
    # Ideally val set should have val_transform (no aug).
    # We will ignore this nuance for the "Beginner-friendly" constraint unless it's critical.
    # It IS critical for validity.
    # Let's create two dataset instances.
    train_ds = BrainMRIDataset(root_dir, transform=get_train_transform())
    val_ds = BrainMRIDataset(root_dir, transform=get_val_transform())
    
    # We need to ensure the split is consistent. `random_split` works on indices.
    # We can use the indices from `random_split` to create Subsets, but we want different transforms.
    # A common trick:
    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index] # This calls original dataset getitem
            # Original dataset applies its transform. 
            # So we should init original dataset with NO transform.
            return x, y
        def __len__(self):
            return len(self.subset)
            
    # Correct way:
    ds_raw = BrainMRIDataset(root_dir, transform=None)
    train_subset, val_subset = random_split(ds_raw, [train_size, val_size])
    
    # Now wrap them to apply transforms
    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            # The base dataset returns (image, label) where image is PIL because transform=None
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_data = TransformWrapper(train_subset, get_train_transform())
    val_data = TransformWrapper(val_subset, get_val_transform())

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    encoder = get_encoder()
    # Load pretrained weights if available
    if os.path.exists("models/encoder.pth"):
        print("Loading SSL pretrained encoder...")
        encoder.load_state_dict(torch.load("models/encoder.pth", weights_only=True))
    else:
        print("No pretrained encoder found. Training from scratch.")

    model = get_brain_classifier(encoder, num_classes=4).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting Supervised Training (Brain MRI)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
    metrics = calculate_metrics(all_labels, all_preds)
    print("Validation Metrics:", metrics)

    # Save Model
    torch.save(model.state_dict(), "models/brain_model.pth")
    print("Brain Classifier saved to models/brain_model.pth")

if __name__ == "__main__":
    train_brain()
