import os
import gdown
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

MODEL_PATH = "model.pth"
FILE_ID = "19X17M9QMTrbWMhoxRvDh_Lfuk4aURfi-"

device = torch.device("cpu")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    gdown.download(url=f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing", output=MODEL_PATH, quiet=False, fuzzy=True)
    print("Download done!")

print("Loading model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img):
    if img is None:
        return "Upload an image"
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).mean().item()
    label = "FAKE 🔴" if score > 0.5 else "REAL 🟢"
    return f"{label} - Score: {score:.4f}"

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="DeepGuard Lite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
