import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

device = torch.device("cpu")
print("Loading local model.pth...")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256)

state = torch.load("model.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img):
    if img is None:
        return "Upload image"
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).mean().item()
    return f"{'FAKE 🔴' if score > 0.5 else 'REAL 🟢'} - {score:.4f}"

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="DeepGuard Lite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
