import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from models.classifier import get_brain_classifier, get_knee_classifier
from utils.augmentations import get_val_transform

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load Models (ONCE)
# -----------------------------
print("Loading models...")

brain_model = get_brain_classifier(None, num_classes=4)
brain_model.load_state_dict(
    torch.load("models/brain_model.pth", map_location=device)
)
brain_model.to(device).eval()

knee_model = get_knee_classifier(None, num_classes=5)
knee_model.load_state_dict(
    torch.load("models/knee_model.pth", map_location=device)
)
knee_model.to(device).eval()

print("Models loaded successfully.")

# -----------------------------
# Labels & Transform
# -----------------------------
BRAIN_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
KNEE_CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

transform = get_val_transform()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image, task):
    if image is None:
        return {}

    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if task == "Brain MRI":
            outputs = brain_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            return {BRAIN_CLASSES[i]: float(probs[i]) for i in range(len(BRAIN_CLASSES))}
        else:
            outputs = knee_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            return {KNEE_CLASSES[i]: float(probs[i]) for i in range(len(KNEE_CLASSES))}

# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Medical Image"),
        gr.Radio(
            ["Brain MRI", "Knee Osteoarthritis"],
            label="Select Task",
            value="Brain MRI"
        )
    ],
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="SSCLNet Medical Image Analysis",
    description=(
        "Upload a Brain MRI or Knee X-ray image to get "
        "classification results using SSCLNet."
    )
)

iface.launch()
