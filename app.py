import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import gdown
import cv2
import numpy as np

# ===== CONFIG - FINAL =====
MODEL_PATH = "deepguard_lite_c40.pth"
FILE_ID = "19X17M9QMTrbWMhoxRvDh_Lfuk4aURfi-"
DEVICE = torch.device("cpu")

# ===== DOWNLOAD FROM DRIVE =====
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("Model downloaded!")

# ===== MODEL ARCHITECTURE =====
def get_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    )
    return model

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===== PREDICTION =====
def predict_file(file):
    if file is None:
        return "Please upload a file"

    # VIDEO
    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        cap = cv2.VideoCapture(file)
        probs = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return "Invalid video"
        for idx in np.linspace(0, total_frames-1, 15, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = torch.sigmoid(model(tensor)).item()
                probs.append(p)
        cap.release()
        if not probs:
            return "Could not read video"
        avg_prob = sum(probs) / len(probs)
        return f"{'FAKE' if avg_prob > 0.5 else 'REAL'} - {avg_prob*100:.1f}% confidence (Fake prob)"

    # IMAGE
    else:
        img = Image.open(file).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        if prob > 0.5:
            return f"FAKE - {prob*100:.1f}% confidence"
        else:
            return f"REAL - {(1-prob)*100:.1f}% confidence"

# ===== GRADIO UI =====
with gr.Blocks(title="DeepGuard Lite") as demo:
    gr.Markdown("# DeepGuard Lite - C40 Deepfake Detector")
    gr.Markdown("Upload image or video (works for low quality / compressed)")
    file_input = gr.File(label="Upload Image/Video", type="filepath")
    result = gr.Textbox(label="Result")
    btn = gr.Button("Detect")
    btn.click(fn=predict_file, inputs=file_input, outputs=result)

demo.launch(server_name="0.0.0.0", server_port=10000)
