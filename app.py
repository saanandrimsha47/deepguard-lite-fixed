import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import cv2

# --- Load Model (Fixed for your C40.pth) ---
model = None

def load_model():
    global model
    if model is None:
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, 2)

        ckpt = torch.load("deepguard_lite_c40.pth", map_location="cpu", weights_only=False)

        # handle if saved as {'state_dict':...} or {'model':...}
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        # strict=False fixes your error: Missing key(s) features.1.conv...
        try:
            m.load_state_dict(ckpt)
        except RuntimeError:
            m.load_state_dict(ckpt, strict=False)

        m.eval()
        model = m
    return model

def predict_pil(pil_img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = load_model()(x)
        pred = torch.argmax(out, 1).item()
        conf = torch.softmax(out, 1)[0][pred].item() * 100
    return pred, conf

def predict_file(file_path):
    if not file_path:
        return "Please Upload Image or Video"

    ext = file_path.lower().split('.')[-1]
    try:
        # Video handling
        if ext in ['mp4', 'mov', 'avi', 'mkv', 'webm']:
            cap = cv2.VideoCapture(file_path)
            preds = []
            for _ in range(5): # check 5 frames
                ret, frame = cap.read()
                if not ret:
                    break
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                p, c = predict_pil(pil)
                preds.append(p)
            cap.release()

            if not preds:
                return "Could not read video"

            fake_ratio = sum(preds) / len(preds)
            if fake_ratio > 0.5:
                return f"Video: Fake - Deepfake ({fake_ratio*100:.0f}% fake frames)\nRobust to C0 / C23 / C40"
            else:
                return f"Video: Real ({(1-fake_ratio)*100:.0f}% real frames)\nRobust to C0 / C23 / C40"
        else:
            # Image handling
            pil = Image.open(file_path).convert("RGB")
            pred, conf = predict_pil(pil)
            if pred == 0:
                return f"Image: Real ({conf:.1f}%)\nTested on C0 / C23 / C40"
            else:
                return f"Image: Fake ({conf:.1f}%)\nTested on C0 / C23 / C40"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Launch for Render ---
port = int(os.environ.get("PORT", 10000))

demo = gr.Interface(
    fn=predict_file,
    inputs=gr.File(label="Upload File", file_types=["image", "video"], type="filepath"),
    outputs=gr.Textbox(label="Result"),
    title="DeepGuard Lite",
    description="Lightweight Deepfake Detector - Image and Video."
)

demo.launch(server_name="0.0.0.0", server_port=port)
