# SSCLNet Inference System - Setup & Run Guide

This guide details how to set up the environment and run the **offline** medical image inference system.

## 1. Prerequisites
- **Python 3.9+** installed.
- **cmd** or **PowerShell** access.

## 2. Setup Virtual Environment (Recommended)
It is best practice to use a virtual environment to avoid conflicts.

Open a terminal in the `SSCLNet_Project` folder and run:

```bash
# Create virtual environment (Windows)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

*(You should see `(venv)` appear at the start of your command line)*

## 3. Install Dependencies
Install the required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: This includes PyTorch (CPU version by default), Gradio, and Pillow.*

## 4. Model Setup
Ensure your trained model files are placed correctly:
- `models/brain_model.pth`
- `models/knee_model.pth`
- `models/encoder.pth` (if needed for retraining, though inference uses the full models above)

## 5. Run the Application
Start the offline inference server:

```bash
python app.py
```

Wait a few seconds. You will see a message like:
`Running on local URL:  http://127.0.0.1:7860`

The browser should open automatically. If not, copy and paste the link into Chrome/Edge.

## 6. How to Use
1. **Title**: "SSCLNet Medical Image Analysis"
2. **Upload**: Click to upload an image from your computer (Brain MRI or Knee X-Ray).
3. **Select Task**: Choose "Brain MRI" or "Knee Osteoarthritis" via the radio buttons.
4. **Result**: The system displays the predicted Class and Confidence score.

## Troubleshooting
- **"Module not found" error**: Ensure you activated the `venv` and installed requirements.
- **Browser doesn't open**: Manually visit `http://127.0.0.1:7860`.
- **System is slow**: First run might be slower as models initialize; subsequent runs are instant.
