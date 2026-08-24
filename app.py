import os, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import cv2

model = None
def get_model():
    global model
    if model is None:
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,2))
        ckpt = torch.load("deepguard_lite_c40.pth", map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ["state_dict","model"]:
                if k in ckpt: ckpt = ckpt[k]; break
        m.load_state_dict(ckpt, strict=False)
        m.eval()
        model = m
    return model

def predict_file(path):
    get_model()
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    def pred(pil):
        x=tf(pil).unsqueeze(0)
        with torch.no_grad():
            out=model(x); prob=torch.softmax(out,1); p=torch.argmax(prob,1).item(); c=prob[0][p].item()*100
        return p,c
    ext=path.split('.')[-1].lower()
    if ext in ['mp4','mov','avi','mkv','webm']:
        cap=cv2.VideoCapture(path); s=[]
        for _ in range(15):
            ret,fr=cap.read()
            if not ret: break
            s.append(pred(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))[0])
        cap.release()
        return f'{"FAKE" if sum(s)/len(s)>0.5 else "REAL"} - {sum(s)/len(s)*100:.1f}%'
    else:
        p,c=pred(Image.open(path).convert("RGB"))
        return f'{"FAKE" if p==1 else "REAL"} - {c:.1f}%'

gr.Interface(fn=predict_file, inputs=gr.File(type="filepath"), outputs=gr.Textbox(label="Result"), title="DeepGuard Lite").launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT",10000)))
