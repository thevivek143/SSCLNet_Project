import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys
import numpy as np
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor

# ==========================================
# PATH SETUP
# ==========================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "knee"))

# ==========================================
# IMPORTS (BRAIN – UNCHANGED)
# ==========================================
from models.classifier import get_brain_classifier
from models.encoder import get_encoder
from utils.augmentations import get_val_transform
from utils.grad_cam import GradCAM, overlay_heatmap

# ==========================================
# IMPORTS (KNEE – NEW 3-CLASS PIPELINE)
# ==========================================
from knee.models.encoder import DenseNetEncoder
from knee.models.classifier import KneeClassifier
from knee.utils.augmentations import get_val_transform as get_knee_transform

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cpu")

BRAIN_MODEL_PATH = os.path.join("models", "brain_model.pth")
KNEE_MODEL_PATH = os.path.join("knee", "models", "best_knee_model.pth")

BRAIN_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 🔥 UPDATED 3-CLASS OA LABELS
KNEE_CLASSES = [
    "No / Doubtful OA",
    "Mild–Moderate OA",
    "Severe OA"
]

print(f"System running on {DEVICE}")

# ==========================================
# ROI CROP (KNEE – SAME AS TRAINING)
# ==========================================
def knee_roi_crop(pil_image):
    img = np.array(pil_image)
    h, w, _ = img.shape
    img = img[int(0.35*h):int(0.75*h), int(0.2*w):int(0.8*w)]
    return Image.fromarray(img)

# ==========================================
# PDF REPORT GENERATION
# ==========================================
def generate_report(image, heatmap, diagnosis_text, probability_text):
    if image is None:
        return None

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/SSCLNet_Report_{timestamp}.pdf"

    temp_img = f"reports/tmp_img_{timestamp}.jpg"
    temp_cam = f"reports/tmp_cam_{timestamp}.jpg"

    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_img)
    else:
        image.save(temp_img)

    if heatmap is not None:
        if isinstance(heatmap, np.ndarray):
            Image.fromarray(heatmap).save(temp_cam)
        else:
            heatmap.save(temp_cam)

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(HexColor("#0f766e")) # Teal color
    c.rect(0, height - 100, width, 100, fill=1, stroke=0)
    
    c.setFont("Helvetica-Bold", 28)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(50, height - 60, "SSCLNet Diagnostic Report")

    # Content
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#333333"))
    c.drawString(50, height - 150, "Diagnostic Results")
    
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 180, f"Predicted Diagnosis: {diagnosis_text}")
    c.drawString(50, height - 205, f"Model Confidence: {probability_text}")

    # Images
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 260, "Original Scan")
    c.drawImage(temp_img, 50, height - 470, width=200, height=200)

    if os.path.exists(temp_cam):
        c.drawString(300, height - 260, "AI Analysis (Grad-CAM)")
        c.drawImage(temp_cam, 300, height - 470, width=200, height=200)

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(HexColor("#666666"))
    c.drawString(50, 40, f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 25, "Note: This is an AI-assisted tool and should not replace professional medical advice.")

    c.save()

    os.remove(temp_img)
    if os.path.exists(temp_cam):
        os.remove(temp_cam)

    return filename

# ==========================================
# LOAD MODELS
# ==========================================
def load_models():
    # ---------- Brain ----------
    brain_encoder = get_encoder(pretrained=False)
    brain_model = get_brain_classifier(brain_encoder, len(BRAIN_CLASSES))
    if os.path.exists(BRAIN_MODEL_PATH):
        try:
            brain_model.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location=DEVICE))
            print("Loaded Brain MRI model.")
        except Exception as e:
            print(f"Error loading Brain MRI model: {e}")
    brain_model.to(DEVICE).eval()
    brain_cam = GradCAM(brain_model, brain_model.encoder[7])

    # ---------- Knee (NEW 3-CLASS) ----------
    knee_encoder = DenseNetEncoder().backbone
    knee_model = KneeClassifier(knee_encoder)
    if os.path.exists(KNEE_MODEL_PATH):
        try:
            knee_model.load_state_dict(torch.load(KNEE_MODEL_PATH, map_location=DEVICE))
            print("Loaded Knee OA model.")
        except Exception as e:
            print(f"Error loading Knee OA model: {e}")
    knee_model.to(DEVICE).eval()
    knee_cam = GradCAM(knee_model, knee_model.encoder.features.denseblock4)

    return (brain_model, brain_cam), (knee_model, knee_cam)

(brain_model, brain_cam), (knee_model, knee_cam) = load_models()
brain_transform = get_val_transform()
knee_transform = get_knee_transform()

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict(image, task, cam_alpha):
    if image is None:
        return None, None, "", "", ""

    pil_img = Image.fromarray(image).convert("RGB")

    if task == "Knee Osteoarthritis":
        # ROI Crop for prediction
        roi_img = knee_roi_crop(pil_img)
        img_tensor = knee_transform(image=np.array(roi_img))["image"].unsqueeze(0).to(DEVICE)
        model = knee_model
        cam_tool = knee_cam
        classes = KNEE_CLASSES
        # For visualization, we use the ROI cropped image
        vis_img = roi_img
    else:
        img_tensor = brain_transform(pil_img).unsqueeze(0).to(DEVICE)
        model = brain_model
        cam_tool = brain_cam
        classes = BRAIN_CLASSES
        vis_img = pil_img

    with torch.enable_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].detach().cpu().numpy()
        idx = np.argmax(probs)

        heatmap = cam_tool(img_tensor, class_idx=idx)
        # Apply alpha from slider
        cam_img = overlay_heatmap(heatmap, vis_img, alpha=cam_alpha)
        
        # Upscale for better visibility in UI
        if isinstance(cam_img, np.ndarray):
            cam_img = Image.fromarray(cam_img)
        w, h = cam_img.size
        new_w = 600
        new_h = int(new_w * h / w)
        cam_img = cam_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        label_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        # Color-coded diagnosis
        color = "#ef4444" if "Severe" in classes[idx] or "Glioma" in classes[idx] or "Meningioma" in classes[idx] or "Pituitary" in classes[idx] else "#22c55e"
        if "Mild" in classes[idx]: color = "#eab308"
        
        html = f"""
        <div style="background-color: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center;">
            <h2 style="color: {color}; margin: 0; font-size: 24px;">{classes[idx]}</h2>
            <p style="color: #666; margin: 5px 0 0 0; font-size: 16px;">Confidence: <strong>{probs[idx]*100:.1f}%</strong></p>
        </div>
        """

        return label_dict, cam_img, html, classes[idx], f"{probs[idx]*100:.2f}%"

# ==========================================
# GRADIO UI
# ==========================================
def create_app():
    # Use a professional medical theme
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_text_color="white",
    )

    with gr.Blocks(theme=theme, title="SSCLNet Medical AI") as app:
        diagnosis_state = gr.State()
        confidence_state = gr.State()

        # Header
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1>🏥 SSCLNet Medical Analysis</h1>
                        <p style="font-size: 1.2em;">Advanced Self-Supervised Learning for <b>Brain Tumor</b> & <b>Knee Osteoarthritis</b> Detection</p>
                    </div>
                    """
                )
           
        # Main Layout
        with gr.Row():
            # LEFT COLUMN: INPUTS
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 1. Configuration")
                task = gr.Radio(
                    ["Brain MRI", "Knee Osteoarthritis"],
                    value="Brain MRI",
                    label="Select Diagnostic Task",
                    info="Choose the type of medical scan you are analyzing."
                )
                
                gr.Markdown("### 2. Upload Scan")
                image_input = gr.Image(type="numpy", label="Input Image", height=300)
                
                with gr.Accordion("Advanced Options", open=False):
                    alpha_slider = gr.Slider(0, 1, value=0.4, label="Heatmap Opacity", info="Adjust transparency of the AI attention overlay.")

                run_btn = gr.Button("🔍 Run Analysis", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
                
                gr.Markdown("### 📝 Export")
                report_btn = gr.Button("📄 Download PDF Report", variant="secondary")
                pdf_out = gr.File(label="Generated Report")

            # RIGHT COLUMN: RESULTS
            with gr.Column(scale=2):
                gr.Markdown("### 🧠 Diagnostic Results")
                
                html_out = gr.HTML(label="Diagnosis")
                
                gr.Markdown("### 🔎 HeatMap (Grad-CAM)")
                cam_out = gr.Image(label="Visualization", interactive=False)
                
                gr.Markdown("### 📊 Confidence Scores")
                probs_out = gr.Label(num_top_classes=4, label="Class Probabilities")

        # Example Images
        with gr.Row():
            gr.Examples(
                examples=[
                    ["datasets/brain_mri/glioma/glioma1.png", "Brain MRI"],
                    ["datasets/knee_oa/test/0/9003175L.png", "Knee Osteoarthritis"],
                ],
                inputs=[image_input, task],
                label="Click an example to test:"
            )

        # Instructions
        with gr.Accordion("ℹ️ How to use this tool", open=False):
            gr.Markdown("""
            1. **Select Task**: Choose between Brain MRI or Knee OA analysis.
            2. **Upload Image**: Drag and drop a medical scan (JPG/PNG).
            3. **Run Analysis**: Click the button to generate a diagnosis and heatmap.
            4. **Interpret**: 
               - The **Diagnosis** box shows the top prediction.
               - The **Grad-CAM** image highlights the area the AI focused on.
               - The **Confidence** scores show the model's certainty.
            5. **Report**: Download a PDF summary for records.
            """)

        # Event Listeners
        run_btn.click(
            predict,
            inputs=[image_input, task, alpha_slider],
            outputs=[probs_out, cam_out, html_out, diagnosis_state, confidence_state]
        )

        report_btn.click(
            generate_report,
            inputs=[image_input, cam_out, diagnosis_state, confidence_state],
            outputs=pdf_out
        )
        
        def reset_app():
            return None, None, None, "", None, None, None

        reset_btn.click(
            reset_app,
            inputs=[],
            outputs=[image_input, probs_out, cam_out, html_out, diagnosis_state, confidence_state, pdf_out]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
