# SSCLNet: Self-Supervised Contrastive Learning Framework

## Overview
SSCLNet is a medical image classification framework that leverages **Self-Supervised Contrastive Learning (SSCL)** using the SimCLR approach. The system extracts robust features from Brain MRI scans and uses them to classify Brain Tumors. It also features a supervised classifier for Knee Osteoarthritis severity grading (Kellgren–Lawrence grade).

## Project Structure
```
SSCLNet_Project/
├── datasets/               # Data files
│   ├── brain_mri/          # 4 classes: glioma, meningioma, pituitary, no_tumor
│   └── knee_oa/            # 5 classes: 0-4
├── models/                 # Neural Network architectures
│   ├── encoder.py          # ResNet-18 Backbone
│   ├── projection_head.py  # MLP for Contrastive Learning
│   └── classifier.py       # Classification Heads
├── utils/                  # Helper scripts
│   ├── augmentations.py    # SimCLR and Supervisor transforms
│   ├── dataset.py          # Custom PyTorch Datasets
│   └── metrics.py          # Evaluation metrics
├── train_ssl.py            # Step 1: Train Encoder (Self-Supervised)
├── train_brain_classifier.py # Step 2: Train Brain MRI Classifier
├── train_knee_classifier.py  # Step 3: Train Knee OA Classifier
├── evaluate.py             # Evaluation script
├── app.py                  # Streamlit Web Interface
└── requirements.txt        # Python dependencies
```

## Installation
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Pipeline

### Step 1: Self-Supervised Learning (SimCLR)
Train the encoder on Brain MRI data using contrastive learning. This saves `models/encoder.pth`.
```bash
python train_ssl.py
```

### Step 2: Train Brain MRI Classifier
Fine-tune the pretrained encoder (or train on top of it) for Brain Tumor classification. Saves `models/brain_model.pth`.
```bash
python train_brain_classifier.py
```

### Step 3: Train Knee OA Classifier
Train the model for Knee Osteoarthritis grading. Saves `models/knee_model.pth`.
```bash
python train_knee_classifier.py
```

## Evaluation
Run the evaluation script to see metrics (Accuracy, Precision, Recall, F1) on the validation set.
```bash
python evaluate.py
```

## Run the Web Application
Launch the Streamlit interface to predict on new images.
```bash
streamlit run app.py
```
else
```
# 1. Install Gradio
!pip install gradio

# 2. Create the App
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from models.encoder import get_encoder
from models.classifier import get_brain_classifier, get_knee_classifier
from utils.augmentations import get_val_transform
import os

# Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Encoder
encoder = get_encoder()
if os.path.exists("models/encoder.pth"):
    encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device, weights_only=True))

# Brain Model
brain_model = get_brain_classifier(get_encoder(), num_classes=4)
if os.path.exists("models/brain_model.pth"):
    brain_model.load_state_dict(torch.load("models/brain_model.pth", map_location=device, weights_only=True))
brain_model.to(device).eval()

# Knee Model
knee_model = get_knee_classifier(get_encoder(), num_classes=5)
if os.path.exists("models/knee_model.pth"):
    knee_model.load_state_dict(torch.load("models/knee_model.pth", map_location=device, weights_only=True))
knee_model.to(device).eval()

# Labels
BRAIN_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
KNEE_CLASSES = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
transform = get_val_transform()

def predict(image, task):
    if image is None:
        return "Please upload an image."
    
    # Preprocess
    img_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if task == "Brain MRI":
            output = brain_model(img_tensor)
            probs = F.softmax(output, dim=1)
            # Create dict for Gradio Label output
            return {BRAIN_CLASSES[i]: float(probs[0][i]) for i in range(len(BRAIN_CLASSES))}
        else:
            output = knee_model(img_tensor)
            probs = F.softmax(output, dim=1)
            return {KNEE_CLASSES[i]: float(probs[0][i]) for i in range(len(KNEE_CLASSES))}

# Create Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Medical Image"),
        gr.Radio(["Brain MRI", "Knee Osteoarthritis"], label="Select Task", value="Brain MRI")
    ],
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="SSCLNet Medical Image Analysis",
    description="Upload a Brain MRI or Knee X-ray to classify it."
)

# Launch with sharing enabled
iface.launch(share=True)

```

## Example Outputs
- **Brain MRI**: Predicts "Glioma", "Meningioma", "Pituitary", or "No Tumor".
- **Knee OA**: Predicts severity Grade 0 to Grade 4.
