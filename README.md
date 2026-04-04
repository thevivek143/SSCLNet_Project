# SSCLNet: Advanced Medical Image Analysis

**SSCLNet** (Self-Supervised Contrastive Learning Network) is a medical AI framework designed for the robust analysis of **Brain MRI** scans and **Knee Osteoarthritis** X-rays. It utilizes Self-Supervised Learning (SimCLR) and Dual-Attentive DenseNet architectures to achieve high diagnostic accuracy with interpretability features like Grad-CAM heatmaps.

## 🚀 Key Features
- **Dual-Task Diagnosis**: Unified interface for Brain Tumor classification (4 classes) and Knee Osteoarthritis grading (3 severity levels).
- **Self-Supervised Learning**: Leverages SimCLR to learn robust feature representations from unlabeled medical data.
- **Explainable AI (XAI)**: Integrated Grad-CAM visualization to show *where* the model is looking.
- **Professional Dashboard**: A clean, medical-grade UI built with Gradio 5.0, featuring dark/light modes, PDF reporting, and side-by-side analysis.
- **PDF Reporting**: Automatically generates downloadable diagnostic reports with heatmaps and confidence scores.

## 📂 Project Structure
```
SSCLNet/
├── app.py                  # 🏥 Main Medical Dashboard (Gradio)
├── requirements.txt        # Python dependencies
├── datasets/               # Training and Test datasets
│   ├── brain_mri/          # (Glioma, Meningioma, Pituitary, No Tumor)
│   └── knee_oa/            # (Grades 0-4 mapped to 3 classes)
├── knee/                   # 🦴 Knee Osteoarthritis Module
│   ├── models/             # DenseNet121 w/ Attention
│   ├── utils/              # ROI Cropping & Augmentations
│   ├── train_knee_classifier.py
│   └── evaluate_knee_classifier.py
├── models/                 # 🧠 Brain MRI Module
│   ├── brain_model.pth     # Trained Brain Classifier
│   ├── encoder.py          # ResNet Backbone
│   └── classifier.py       # Classification Head
└── utils/                  # Shared Utilities
    ├── grad_cam.py         # Visualization logic
    └── augmentations.py    # Image transforms
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thevivek0408/ssclnet.git
   cd ssclnet
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Web Dashboard
Launch the full medical diagnostic interface locally:
```bash
python app.py
```
*The app will open in your browser at `http://127.0.0.1:7860`.*

### Using the App
1. **Select Task**: Choose "Brain MRI" or "Knee Osteoarthritis".
2. **Upload Image**: Drop a medical scan (JPG/PNG).
3. **Run Analysis**: Click to see diagnosis, probability bars, and heatmap.
4. **Download Report**: Generate a PDF summary of the findings.

## 🧠 Model Details

### Brain Tumor Classification
- **Architecture**: ResNet-18 Encoder (Pretrained via SimCLR)
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Method**: Self-Supervised Contrastive Learning on unlabeled MRI data followed by supervised fine-tuning.

### Knee Osteoarthritis Grading
- **Architecture**: DenseNet-121 with Dual-Attention Mechanism
- **Preprocessing**: Automatic ROI Cropping to focus on the knee joint.
- **Classes**: 
  - **No/Doubtful OA** (Grades 0-1)
  - **Mild-Moderate OA** (Grades 2-3)
  - **Severe OA** (Grade 4)

##  training Scripts (Optional)
To retrain the models from scratch:

**1. Self-Supervised Pretraining (Brain):**
```bash
python train_ssl.py
```

**2. Train Brain Classifier:**
```bash
python train_brain_classifier.py
```

**3. Train Knee Classifier:**
```bash
python knee/train_knee_classifier.py
```
