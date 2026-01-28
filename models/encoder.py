import torch
import torch.nn as nn
import torchvision.models as models

def get_encoder(pretrained=False):
    """
    Returns a ResNet-18 encoder without the final classification layer.
    Output dimension is 512.
    """
    # Use weights=None or weights="IMAGENET1K_V1" if pretrained is desired later
    # Project specs say "Self-Supervised Contrastive Learning Framework", implying we train from scratch or init?
    # Usually SSL starts from random or ImageNet. The prompt says "Pretrained encoder weights" are SAVED after SSL.
    # So we likely start with standard initialization (random or ImageNet).
    # Prompt says "Save pretrained encoder weights" -> implies we train it ourselves.
    # Let's default to no pretraining to show purely SSL effect, or ImageNet if "Pretrained" meant ImageNet.
    # "Implement a medical image classification system that uses self-supervised contrastive learning... Save pretrained encoder weights"
    # This implies the "pretrained weights" refers to the ones we get AFTER SSL.
    # So `pretrained=False` (random init) is safer for "showing off" SSL, but `pretrained=True` (ImageNet) is practical.
    # Given "Local execution only... small epochs", starting from ImageNet is much better for convergence.
    # However, to demonstrate SSL, usually one starts random. 
    # Let's stick to random init (`weights=None`) to strictly follow "SSL for extraction" spirit, 
    # but I'll add a comment.
    
    resnet = models.resnet18(weights=None) 
    
    # Remove the fully connected layer
    # We want the output of the avgpool layer (bs, 512, 1, 1) flattened to (bs, 512)
    modules = list(resnet.children())[:-1]
    encoder = nn.Sequential(*modules)
    
    return encoder
