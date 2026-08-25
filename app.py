import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

MODEL_PATH = "deepguard_lite_c40.pth"
device = torch.device("cpu")

print(f"Loading {MODEL_PATH}...")
model = models.efficientnet_b0(weights=None)
in_feat = model.classifier[1].in_features

# FIXED ARCHITECTURE - matches your.pth
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
print("Model loaded! C40 is STRONGEST!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_frame(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    return score

def predict_image(img):
    if img is None: return "Upload an image"
    score = predict_frame(img)
    label = "FAKE 🔴" if score > 0.5 else "REAL 🟢"
    return f"{label} - Confidence: {score:.4f}"

def predict_video(video_path):
    if video_path is None: return "Upload a video"
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 20)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            scores.append(predict_frame(pil_img))
            if len(scores) >= 20: break
        idx += 1
    cap.release()
    if not scores: return "Could not read video"
    avg_score = sum(scores) / len(scores)
    fake_count = sum(1 for s in scores if s > 0.5)
    label = "FAKE 🔴" if avg_score > 0.5 else "REAL 🟢"
    return f"{label}\nAvg Score: {avg_score:.4f}\nFake Frames: {fake_count}/{len(scores)}"

with gr.Blocks(title="DeepGuard Lite C40") as demo:
    gr.Markdown("# DeepGuard Lite C40 - Strongest for Blurry/Compressed Fakes")
    with gr.Tab("Image Detector"):
        img_in = gr.Image(type="pil", label="Upload Image")
        img_out = gr.Textbox(label="Result")
        gr.Button("Detect").click(predict_image, inputs=img_in, outputs=img_out)
    with gr.Tab("Video Detector"):
        vid_in = gr.Video(label="Upload Video")
        vid_out = gr.Textbox(label="Result")
        gr.Button("Detect").click(predict_video, inputs=vid_in, outputs=vid_out)

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
