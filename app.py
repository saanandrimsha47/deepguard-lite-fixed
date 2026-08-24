import os
import gradio as gr

def predict(img):
    return "Working! Model will be added next"

port = int(os.environ.get("PORT", 10000))
demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Textbox(), title="DeepGuard Lite")
demo.launch(server_name="0.0.0.0", server_port=port)
