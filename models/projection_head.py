import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection Head for Contrastive Learning (SimCLR).
    Projects 512-dim embedding to lower dimension (e.g., 128).
    Structure: Generic MLP (Linear -> ReLU -> Linear).
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Flatten input if it comes from GlobalAvgPool (bs, 512, 1, 1)
        x = x.view(x.size(0), -1)
        return self.net(x)
