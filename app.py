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
print("Model loaded! C40 is STRONGEST!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- GRAD-CAM FOR PYTORCH ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        score = torch.sigmoid(out)
        # we want gradient for FAKE class, so backprop the logit itself
        out.backward()

        # pooled gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).detach().cpu()
        heatmap = torch.maximum(heatmap, torch.tensor(0))
        heatmap /= torch.max(heatmap) + 1e-8
        return heatmap.numpy(), score.item()

# Target layer is last conv in efficientnet_b0
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

def predict_frame_with_heatmap(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    heatmap, score = gradcam(x)
    return score, heatmap

def predict_frame(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    return score

def apply_colormap_on_image(org_im, heatmap):
    # org_im: PIL image
    org_im = np.array(org_im.resize((224,224)))
    heatmap = cv2.resize(heatmap, (org_im.shape[1], org_im.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(org_im, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed)

def predict_image(img):
    if img is None: return "Upload an image", None
    score, heatmap = predict_frame_with_heatmap(img)
    label = "FAKE 🔴" if score > 0.41 else "REAL 🟢"
    heatmap_img = apply_colormap_on_image(img, heatmap)
    return f"{label} - Confidence: {score:.4f}", heatmap_img

def predict_video(video_path):
    if video_path is None: return "Upload a video", None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return "Could not read video", None

    frame_indices = [int(total_frames * i / 5) for i in range(5)]
    scores = []
    heatmaps = []
    frames_for_heatmap = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        frame_resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        score, heatmap = predict_frame_with_heatmap(pil_img)
        scores.append(score)
        heatmaps.append(heatmap)
        frames_for_heatmap.append(pil_img)

    cap.release()
    if not scores: return "Could not read video", None
    avg_score = sum(scores) / len(scores)
    # Show heatmap of most fake frame
    max_idx = int(np.argmax(scores))
    best_heatmap_img = apply_colormap_on_image(frames_for_heatmap[max_idx], heatmaps[max_idx])

    fake_count = sum(1 for s in scores if s > 0.41)
    label = "FAKE 🔴" if avg_score > 0.41 else "REAL 🟢"
    text = f"{label}\nAvg Score: {avg_score:.4f}\nFake Frames: {fake_count}/{len(scores)}\n(Checked {len(scores)} key frames)"
    return text, best_heatmap_img

with gr.Blocks(title="DeepGuard Lite C40") as demo:
    gr.Markdown("# DeepGuard Lite C40 - With Grad-CAM Explainability")
    with gr.Tab("Image Detector"):
        img_in = gr.Image(type="pil", label="Upload Image")
        with gr.Row():
            img_out = gr.Textbox(label="Result")
            heatmap_out = gr.Image(type="pil", label="Grad-CAM Heatmap - Red = Fake Region")
        gr.Button("Detect").click(predict_image, inputs=img_in, outputs=[img_out, heatmap_out])
    with gr.Tab("Video Detector"):
        vid_in = gr.Video(label="Upload Video")
        with gr.Row():
            vid_out = gr.Textbox(label="Result")
            vid_heatmap_out = gr.Image(type="pil", label="Most Suspicious Frame + Heatmap")
        gr.Button("Detect").click(predict_video, inputs=vid_in, outputs=[vid_out, vid_heatmap_out])

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
