import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation for SSCLNet.
    Target: The last convolutional layer of the ResNet encoder (layer4).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are a tuple, take the first one
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Compute Grad-CAM heatmap.
        x: Input tensor (1, C, H, W)
        class_idx: Target class index. If None, uses the highest predicted class.
        """
        # 1. Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # 3. Generate Heatmap
        # Global Average Pooling of gradients (weights)
        gradients = self.gradients[0] # (C, 7, 7)
        pooled_gradients = torch.mean(gradients, dim=[1, 2]) # (C,)
        
        # Weight the activations
        activations = self.activations[0] # (C, 7, 7)
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the weighted activations (heatmap)
        heatmap = torch.mean(activations, dim=0).cpu().detach()
        
        # Apply ReLU (we only care about positive influence)
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap to 0-1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        else:
            heatmap = heatmap  # Keep as zeros if everything is zero
            
        return heatmap.numpy()

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """
    Overlays the heatmap on the original PIL image.
    """
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    
    # Convert heatmap to RGB (color map)
    # Use uint8 for color mapping
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert original image to numpy
    original_np = np.array(original_image)
    
    # Blend
    overlaid_img = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid_img
