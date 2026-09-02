import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np

MODEL_PATH = "deepguard_lite_c40.pth"
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count())

print(f"Loading {MODEL_PATH}...")
model = models.efficientnet_b0(weights=None)
in_feat = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_feat, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(256, 1)
)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

print("Warming up...")
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model(dummy)
print("Warm! Model ready.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(img):
    if img is None:
        return "Upload an image"
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    label = "FAKE 🔴" if score > 0.41 else "REAL 🟢"
    return f"{label} - Confidence: {score:.4f}"


def predict_video(video_path):
    if video_path is None:
        return "Upload a video"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return "Could not read video"

    frame_indices = set(int(total_frames * i / 5) for i in range(5))
    max_target = max(frame_indices)

    scores = []
    idx = 0
    while idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frame = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            with torch.no_grad():
                s = torch.sigmoid(model(transform(pil_img).unsqueeze(0).to(device))).item()
            scores.append(s)
        idx += 1
    cap.release()

    if not scores:
        return "Could not read video"

    avg_score = sum(scores) / len(scores)
    fake_count = sum(1 for s in scores if s > 0.41)
    label = "FAKE 🔴" if avg_score > 0.41 else "REAL 🟢"

    return f"{label}\nAvg Score: {avg_score:.4f}\nFake Frames: {fake_count}/{len(scores)}"


with gr.Blocks(title="DeepGuard Lite - Strongest for Blurry/Compressed Fakes") as demo:
    gr.Markdown("# DeepGuard Lite - Strongest for Blurry/Compressed Fakes")
    gr.Markdown("### C40 Model - Fast CPU Inference")

    with gr.Tab("Image Detector"):
        img_in = gr.Image(type="pil", label="Upload Image")
        img_out = gr.Textbox(label="Result")
        gr.Button("Detect").click(predict_image, inputs=img_in, outputs=img_out)

    with gr.Tab("Video Detector"):
        vid_in = gr.Video(label="Upload Video")
        vid_out = gr.Textbox(label="Result")
        gr.Button("Detect").click(predict_video, inputs=vid_in, outputs=vid_out)

port = int(os.environ.get("PORT", 7860))
demo.queue().launch(server_name="0.0.0.0", server_port=port)
