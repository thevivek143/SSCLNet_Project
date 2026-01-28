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

## Example Outputs
- **Brain MRI**: Predicts "Glioma", "Meningioma", "Pituitary", or "No Tumor".
- **Knee OA**: Predicts severity Grade 0 to Grade 4.
