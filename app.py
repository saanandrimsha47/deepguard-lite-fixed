import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class DeepGuardLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

device = torch.device("cpu")
model = DeepGuardLite()
print("Loading model...")
model.load_state_dict(torch.load("deepguard_lite_c40.pth", map_location=device))
model.eval()
print("Model loaded!")

transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

def predict(img):
    if img is None: return "Please upload an image"
    img = Image.fromarray(img).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        _, pred = torch.max(model(img), 1)
        return "✅ REAL Image Detected" if pred.item()==0 else "🚨 FAKE / Deepfake Detected"

demo = gr.Interface(fn=predict, inputs=gr.Image(label="Upload Image"), outputs=gr.Textbox(label="Result"), title="DeepGuard Lite - c40 Blurry Detector")
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
