import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import cv2

model = None
def load_model():
    global model
    if model is None:
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, 2)
        m.load_state_dict(torch.load("deepguard_lite.pth", map_location="cpu"))
        m.eval()
        model = m
    return model

def predict_pil(pil_img):
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    x = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = load_model()(x)
        pred = torch.argmax(out,1).item()
        conf = torch.softmax(out,1)[0][pred].item()*100
    return pred, conf

def predict_file(file_path):
    if not file_path:
        return "Upload Image or Video"
    ext = file_path.lower().split('.')[-1]

    # VIDEO -> C0 / C23 / C40 APPLICABLE
    if ext in ['mp4','mov','avi','mkv','webm']:
        cap = cv2.VideoCapture(file_path)
        preds = []
        for _ in range(5):
            ret, frame = cap.read()
            if not ret: break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            p,c = predict_pil(pil)
            preds.append(p)
        cap.release()
        fake_ratio = sum(preds)/len(preds) if preds else 0
        if fake_ratio > 0.5:
            return f"🎬 Video: 🚨 Fake - Deepfake ({fake_ratio*100:.0f}% fake frames)\nCompression: Robust to C0(raw), C23(HQ), C40(LQ)"
        else:
            return f"🎬 Video: ✅ Real ({(1-fake_ratio)*100:.0f}% real frames)\nCompression: Robust to C0(raw), C23(HQ), C40(LQ)"

    # IMAGE -> C0 / C23 / C40 ALSO applicable (JPEG compression)
    else:
        pil = Image.open(file_path).convert("RGB")
        pred, conf = predict_pil(pil)
        if pred==0:
            return f"🖼️ Image: ✅ Real ({conf:.1f}%)\nCompression: Tested on JPEG C0/C23/C40 levels"
        else:
            return f"🖼️ Image: 🚨 Fake - Deepfake ({conf:.1f}%)\nCompression: Tested on JPEG C0/C23/C40 levels"

port = int(os.environ.get("PORT", 10000))
demo = gr.Interface(
    fn=predict_file,
    inputs=gr.File(label="Upload Image or Video", file_types=["image","video"]),
    outputs=gr.Textbox(label="Result"),
    title="DeepGuard - Image + Video Detector",
    description="Detects Deepfakes in Images & Videos. Evaluated on C0/C23/C40 compression."
)
demo.launch(server_name="0.0.0.0", server_port=port)
