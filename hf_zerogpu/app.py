"""ZeroGPU Gradio deployment candidate — uses original Qwen3.5-9B weights.

This file runs only if a Hugging Face ZeroGPU Space is properly configured.
Do not represent import/commit as successful GPU deployment.
"""
from __future__ import annotations

import gradio as gr
import spaces
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-9B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
# ZeroGPU provides virtual CUDA device mapping outside decorated functions.
# Model is intentionally prepared at startup, per HF platform recommendations.
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
model.eval()

@spaces.GPU(size="large",duration=60)
def reply(message, history, max_tokens):
    if not isinstance(message, str) or not (1 <= len(message.strip()) <= 3000):
        return "Enter a text prompt (1–3,000 characters)"
    rows = []
    for item in history[-6:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content")
            if isinstance(content,str) and len(content) <= 3000:
                rows.append({"role":item["role"], "content":[{"type":"text","text":content}]})
    rows.append({"role":"user","content":[{"type":"text","text":message}]})
    inputs = processor.apply_chat_template(
        rows,add_generation_prompt=True,tokenize=True,
        return_dict=True,return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs,max_new_tokens=int(max_tokens),do_sample=False)
    new_ids=output[0][inputs["input_ids"].shape[-1]:]
    text=processor.decode(new_ids,skip_special_tokens=True).strip()
    return text or "(model returned no text)"

demo=gr.ChatInterface(
    fn=reply,
    type="messages",
    additional_inputs=[gr.Slider(16,256,value=96,step=16,label="Max new tokens")],
    title="Music Japan · Open-weight 9B on ZeroGPU",
    description="Public open Qwen3.5 model · shared temporary GPU, daily limits · not Claude Fable/Opus",
)
if __name__=="__main__":
    demo.queue(default_concurrency_limit=1).launch()
