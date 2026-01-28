import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder import get_encoder
from models.classifier import get_knee_classifier
from utils.dataset import KneeOADataset
from utils.augmentations import get_train_transform, get_val_transform
from utils.metrics import calculate_metrics
import os

def train_knee():
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 1
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Dataset
    root_dir = "datasets/knee_oa"
    train_dataset = KneeOADataset(root_dir, split='train', transform=get_train_transform())
    val_dataset = KneeOADataset(root_dir, split='test', transform=get_val_transform())
    
    if len(train_dataset) == 0:
        print("Knee OA dataset not found or empty.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    encoder = get_encoder()
    # Ensure we start fresh or from SSL? Prompt implies SSL is for Brain MRI. 
    # "SSCLNet: A Self-Supervised Contrastive Learning Framework for Brain MRI... and Knee Osteoarthritis Classification"
    # Usually transfer learning applies. Let's load the SSL encoder if user wants, 
    # but prompt distinction implies "Shared encoder backbone... Classification head: Brain... Knee".
    # I will load the encoder state dict if it exists, to show the benefit of the backbone.
    if os.path.exists("models/encoder.pth"):
        print("Loading SSL pretrained encoder (from Brain MRI task)...")
        encoder.load_state_dict(torch.load("models/encoder.pth", weights_only=True))
    
    model = get_knee_classifier(encoder, num_classes=5).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting Supervised Training (Knee OA)...")
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
    print("Test Metrics:", metrics)

    # Save Model
    torch.save(model.state_dict(), "models/knee_model.pth")
    print("Knee Classifier saved to models/knee_model.pth")

if __name__ == "__main__":
    train_knee()
