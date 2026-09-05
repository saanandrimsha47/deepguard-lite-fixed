import gradio as gr
import torch, cv2, tempfile
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

THRESHOLD = 0.41
MODEL_PATH = "deepguard_lite_c40.pth"

device = torch.device("cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

def predict_frame(pil_img):
    t = tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(t)).item()

def predict_image(img):
    if img is None: return "Upload image"
    prob = predict_frame(img)
    label = "FAKE 🔴" if prob > THRESHOLD else "REAL 🟢"
    return f"{label}\nFake Prob: {prob:.4f} | Threshold: {THRESHOLD}"

def predict_video(video_path):
    if video_path is None: return "Upload video"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, 10, dtype=int)
    probs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs.append(predict_frame(pil))
    cap.release()
    if not probs: return "No face frames"
    avg = sum(probs)/len(probs)
    label = "FAKE 🔴" if avg > THRESHOLD else "REAL 🟢"
    return f"{label}\nAvg Fake Prob: {avg:.4f} (from {len(probs)} frames)\nThreshold: {THRESHOLD}\nAll probs: {[f'{p:.2f}' for p in probs]}"

with gr.Blocks() as demo:
    gr.Markdown("# DeepGuard Lite C40 - Strongest for Blurry/Compressed Fakes\n**Fine-tuned EfficientNet-B0**")
    with gr.Tabs():
        with gr.Tab("Image"):
            img_in = gr.Image(type="pil", label="Upload Face Image")
            img_out = gr.Textbox(label="Result")
            gr.Button("Detect").click(predict_image, img_in, img_out)
        with gr.Tab("Video"):
            vid_in = gr.Video(label="Upload Video (C40)")
            vid_out = gr.Textbox(label="Result")
            gr.Button("Detect Video").click(predict_video, vid_in, vid_out)

demo.launch()
