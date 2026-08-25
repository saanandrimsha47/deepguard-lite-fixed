import os
import gdown
import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0

MODEL_PATH = "deepfake_model.pth"
FILE_ID = "19X17M9QMTrbWMhoxRvDh_Lfuk4aURfi-"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download done!")

device = torch.device("cpu")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()
model.to(device)
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def predict(img):
    if img is None:
        return "Upload an image"
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    label = "FAKE" if score > 0.5 else "REAL"
    return f"{label} - Score: {score:.2f}"

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="DeepGuard Lite")

port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
