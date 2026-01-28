import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder import get_encoder
from models.projection_head import ProjectionHead
from utils.dataset import BrainMRIDataset, PretrainingDataset
from utils.augmentations import get_ssl_transform
import os

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent Loss (SimCLR Loss).
    """
    batch_size = z1.size(0)
    # Concatenate representations
    out = torch.cat([z1, z2], dim=0)
    # Cosine similarity matrix
    sim_matrix = torch.mm(out, out.t().contiguous())
    sim_matrix = nn.functional.normalize(sim_matrix, dim=1)
    
    # We want to use cosine similarity, so usually we normalize vectors first
    # Re-impl cleaner:
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    out = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(out, out.t().contiguous()) / temperature
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=out.device).bool()
    
    # Labels: diagonal interactions are positive pairs (i, i+batch_size) and (i+batch_size, i)
    # But sim_matrix is (2N, 2N).
    # Positive for i is i+N. Positive for i+N is i.
    
    # Standard implementation using CrossEntropy
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(out.device)
    
    # Remove self-similarity from the mask calculation logic if using full matrix
    # Simplified approach:
    # For each i in [0, N-1], target is i+N
    # For each i in [N, 2N-1], target is i-N
    
    # Let's use a simpler standard NT-Xent logic often found:
    # similarities between z1 and z2
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    # This is simplified and only considers z1->z2. Symmetric is better.
    
    # Full symmetric:
    logits_aa = torch.mm(z1, z1.t()) / temperature
    logits_bb = torch.mm(z2, z2.t()) / temperature
    logits_ab = torch.mm(z1, z2.t()) / temperature
    logits_ba = torch.mm(z2, z1.t()) / temperature

    mask = torch.eye(batch_size, device=z1.device).bool()
    logits_aa.masked_fill_(mask, -float('inf'))
    logits_bb.masked_fill_(mask, -float('inf'))

    logits_a = torch.cat([logits_ab, logits_aa], dim=1)
    logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    
    target = torch.arange(batch_size, device=z1.device)
    loss_a = nn.CrossEntropyLoss()(logits_a, target)
    loss_b = nn.CrossEntropyLoss()(logits_b, target)
    
    return loss_a + loss_b

def train_ssl():
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Dataset
    root_dir = "datasets/brain_mri"
    if not os.path.exists(root_dir):
        print(f"Dataset path {root_dir} not found!")
        return

    # Use basic image dataset structure then wrap it
    base_dataset = BrainMRIDataset(root_dir, transform=None) 
    # Transforms are applied in PretrainingDataset
    ssl_dataset = PretrainingDataset(base_dataset, transform=get_ssl_transform())
    
    train_loader = DataLoader(ssl_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    encoder = get_encoder().to(DEVICE)
    proj_head = ProjectionHead().to(DEVICE)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=LR)

    encoder.train()
    proj_head.train()

    print("Starting Self-Supervised Learning (SimCLR)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for x1, x2 in train_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            
            optimizer.zero_grad()
            
            z1 = proj_head(encoder(x1))
            z2 = proj_head(encoder(x2))
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # Save Encoder
    torch.save(encoder.state_dict(), "models/encoder.pth")
    print("Encoder weights saved to models/encoder.pth")

if __name__ == "__main__":
    train_ssl()
