import os, traceback
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import cv2

model = None

def get_model():
    global model
    if model is None:
        # Build EXACT architecture as your training
        m = models.mobilenet_v2(weights=None)
        # This matches your.pth file
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)
        )
        ckpt = torch.load("deepguard_lite_c40.pth", map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
            elif "model" in ckpt: ckpt = ckpt["model"]
            elif "model_state_dict" in ckpt: ckpt = ckpt["model_state_dict"]

        m.load_state_dict(ckpt, strict=True)
        m.eval()
        model = m
        print("Model loaded SUCCESS")
    return model

def predict_file(file_path):
    try:
        if not file_path: return "Please upload a file"
        get_model()
        ext = file_path.lower().split('.')[-1]
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        def pred_pil(pil_img):
            x = tf(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                prob = torch.softmax(out, 1)
                pred = torch.argmax(prob, 1).item()
                conf = prob[0][pred].item()*100
            return pred, conf

        if ext in ['mp4','mov','avi','mkv','webm']:
            cap = cv2.VideoCapture(file_path)
            scores = []
            for _ in range(10):
                ret, frame = cap.read()
                if not ret: break
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                p, c = pred_pil(pil)
                scores.append(p)
            cap.release()
            if not scores: return "Could not read video"
            fake_ratio = sum(scores)/len(scores)
            return f"{'FAKE' if fake_ratio>0.5 else 'REAL'} - {fake_ratio*100:.1f}% fake frames"
        else:
            pil = Image.open(file_path).convert("RGB")
            p, c = pred_pil(pil)
            label = "FAKE" if p==1 else "REAL"
            return f"{label} - {c:.1f}% confidence"

    except Exception as e:
        return f"Error: {traceback.format_exc()}"

port = int(os.environ.get("PORT", 10000))
gr.Interface(
    fn=predict_file,
    inputs=gr.File(type="filepath", label="Upload File"),
    outputs=gr.Textbox(label="Result"),
    title="DeepGuard Lite",
    description="Lightweight Deepfake Detector - Image and Video."
).launch(server_name="0.0.0.0", server_port=port)
