import os
import gradio as gr
from PIL import Image

def predict(img):
    return "Model loaded - Fake/Real prediction will show here"

port = int(os.environ.get("PORT", 10000))
print(f"Starting on 0.0.0.0:{port}")

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Textbox(), title="DeepGuard Lite")
demo.launch(server_name="0.0.0.0", server_port=port)
