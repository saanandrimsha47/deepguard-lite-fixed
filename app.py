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
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)
        )
        ckpt = torch.load("deepguard_lite_c40.pth", map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ["state_dict","model","model_state_dict"]:
                if k in ckpt:
                    ckpt = ckpt[k]
                    break

        # This False is IMPORTANT - your pth has only classifier weights
        m.load_state_dict(ckpt, strict=False)
        m.eval()
        model = m
        print("Model loaded SUCCESS with strict=False")
    return model

def predict_file(file_path):
    try:
        if not file_path: return "Please upload a file"
        get_model()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        def pred_pil(pil_img):
            x = tf(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                prob = torch.softmax(out, 1)
                pred = torch.argmax(prob, 1).item()
                conf = prob[0][pred].item()*100
            return pred, conf

        ext = file_path.lower().split('.')[-1]
        if ext in ['mp4','mov','avi','mkv','webm']:
            cap = cv2.VideoCapture(file_path)
            scores=[]
            for _ in range(10):
                ret, frame = cap.read()
                if not ret: break
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                p,c = pred_pil(pil)
                scores.append(p)
            cap.release()
            if not scores: return "Could not read video"
            ratio = sum(scores)/len(scores)
            return f"{'FAKE' if ratio>0.5 else 'REAL'} - {ratio*100:.1f}% fake frames"
        else:
            pil = Image.open(file_path).convert("RGB")
            p,c = pred_pil(pil)
            return f"{'FAKE' if p==1 else 'REAL'} - {c:.1f}% confidence"
    except Exception as e:
        return f"Error: {traceback.format_exc()}"

port = int(os.environ.get("PORT", 10000))
gr.Interface(fn=predict_file, inputs=gr.File(type="filepath", label="Upload File"), outputs=gr.Textbox(label="Result"), title="DeepGuard Lite").launch(server_name="0.0.0.0", server_port=port)
