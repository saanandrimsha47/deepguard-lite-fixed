import os
import gdown
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

MODEL_PATH = "model.pth"
FILE_ID = "19X2g9v2M3r5B7v9aZpLq1wK8xYt0cDeFg" # <-- YOUR NEW FILE ID FROM DRIVE, KEEP YOURS
# If you used my last ID, keep that same one, don't change

device = torch.device("cpu")

# 1. Download
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download done!")

# 2. Build model - FIXED TO 256 to match your checkpoint
print("Loading model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
print("Model loaded successfully!")

# 3. Transform + Predict
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
        # Your model outputs 256 values, we take average as fake score
        score = torch.sigmoid(out).mean().item()
    label = "FAKE 🔴" if score > 0.5 else "REAL 🟢"
    return f"{label} - Score: {score:.4f}"

# 4. Gradio + Render PORT fix
demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="DeepGuard Lite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
