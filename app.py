import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys

# Import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.classifier import get_brain_classifier, get_knee_classifier
from models.encoder import get_encoder
from utils.augmentations import get_val_transform

# ==========================================
# CONFIGURATION & SETUP
# ==========================================

# Force CPU if requested or just auto-detect
DEVICE = torch.device("cpu") 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"System configured to use: {DEVICE}")

# Define Constants
BRAIN_MODEL_PATH = os.path.join("models", "brain_model.pth")
KNEE_MODEL_PATH = os.path.join("models", "knee_model.pth")

BRAIN_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
KNEE_CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

# ==========================================
# MODEL LOADING
# ==========================================

def load_system_models():
    """
    Loads both Brain and Knee models safely.
    Returns: (brain_model, knee_model) or None if failure.
    """
    try:
        print("Initializing models...")
        
        # 1. Load Brain Model
        # Must instantiate a fresh encoder for each model so they have separate weights
        brain_encoder = get_encoder(pretrained=False)
        brain_net = get_brain_classifier(encoder=brain_encoder, num_classes=len(BRAIN_CLASSES))
        
        if os.path.exists(BRAIN_MODEL_PATH):
            state_dict = torch.load(BRAIN_MODEL_PATH, map_location=DEVICE)
            brain_net.load_state_dict(state_dict)
            print("✅ Brain MRI Model loaded successfully.")
        else:
            print(f"⚠️ Warning: {BRAIN_MODEL_PATH} not found. Brain inference will fail.")
        
        brain_net.to(DEVICE)
        brain_net.eval()
        
        # 2. Load Knee Model
        knee_encoder = get_encoder(pretrained=False)
        knee_net = get_knee_classifier(encoder=knee_encoder, num_classes=len(KNEE_CLASSES))
        
        if os.path.exists(KNEE_MODEL_PATH):
            state_dict = torch.load(KNEE_MODEL_PATH, map_location=DEVICE)
            knee_net.load_state_dict(state_dict)
            print("✅ Knee OA Model loaded successfully.")
        else:
            print(f"⚠️ Warning: {KNEE_MODEL_PATH} not found. Knee inference will fail.")
            
        knee_net.to(DEVICE)
        knee_net.eval()
        
        return brain_net, knee_net

    except Exception as e:
        print(f"❌ Critical Error loading models: {e}")
        # traceback.print_exc() # accessing traceback would require import
        return None, None

# Initialize models globally
brain_model, knee_model = load_system_models()
transform = get_val_transform()

# ==========================================
# INFERENCE LOGIC
# ==========================================

def predict(image, task):
    """
    Main inference function for Gradio.
    Args:
        image: Input image (numpy array from Gradio)
        task: Selected task string
    Returns:
        Dictionary of Class -> Probability
    """
    if image is None:
        return None

    try:
        # Preprocess Image
        pil_image = Image.fromarray(image).convert("RGB")
        
        img_tensor = transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if task == "Brain MRI":
                if brain_model is None: 
                    raise gr.Error("Brain Model failed to load. Check server logs.")
                
                outputs = brain_model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                
                return {
                    BRAIN_CLASSES[i]: float(probs[i]) 
                    for i in range(len(BRAIN_CLASSES))
                }
            
            elif task == "Knee Osteoarthritis":
                if knee_model is None:
                     raise gr.Error("Knee Model failed to load. Check server logs.")
                
                outputs = knee_model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                
                return {
                    KNEE_CLASSES[i]: float(probs[i]) 
                    for i in range(len(KNEE_CLASSES))
                }
            
            else:
                raise gr.Error("Invalid Task Selected")

    except Exception as e:
        # Re-raise Gradio errors, wrap others
        if isinstance(e, gr.Error):
            raise e
        raise gr.Error(f"Inference Failed: {str(e)}")

# ==========================================
# UI CONSTRUCTION
# ==========================================

def create_app():
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(
                type="numpy", 
                label="Upload Medical Image (MRI / X-Ray)",
                sources=["upload"] # Restrict to upload only for simplicity, or add "webcam" if needed
            ),
            gr.Radio(
                ["Brain MRI", "Knee Osteoarthritis"],
                label="Select Clinical Task",
                value="Brain MRI",
                info="Choose the specific analysis pipeline for your image."
            )
        ],
        outputs=gr.Label(
            num_top_classes=4, 
            label="Diagnostic Prediction & Confidence"
        ),
        title="SSCLNet: Self-Supervised Clinical Learning Network",
        description="""
        ### 🏥 Medical Image Analysis System
        **Offline-Capable AI Inference powered by PyTorch**
        
        **Instructions:**
        1. **Upload** a Brain MRI scan or Knee X-Ray.
        2. **Select** the appropriate task below.
        3. **View** the AI's diagnostic probability assessment.
        
        *Note: This system runs entirely locally on your machine. No data leaves your device.*
        """,
        theme=gr.themes.Soft(),
        flagging_mode="never"
    )

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("\nStarting SSCLNet Inference Server...")
    print(f"Open your browser at http://127.0.0.1:7860")
    
    app = create_app()
    app.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        share=False,      # False = No public link (offline mode)
        inbrowser=True    # Try to open browser automatically
    )
