import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)
THRESHOLD = 0.41 # FINAL LOCKED
MODEL_PATH = "deepguard_lite_c40.pth"

# Model = EXACT same as training - Simple head
def get_model():
    model = models.efficientnet_b0(weights=None)
    # This matches your checkpoint: classifier = [Dropout, Linear(1280,1)]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model

device = torch.device("cpu")
model = get_model()
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/")
def home():
    return render_template("index.html") if os.path.exists("templates/index.html") else "DeepGuard Lite Live - POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error":"No file"}), 400
    img = Image.open(file.stream).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
    label = "FAKE" if prob > THRESHOLD else "REAL"
    return jsonify({"label": label, "fake_prob": prob, "threshold": THRESHOLD})

# THIS FIXES RENDER PORT ERROR
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
