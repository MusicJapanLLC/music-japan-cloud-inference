---
title: Music Japan Qwen3.5 9B GPU Lab
emoji: "🧪"
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12.12
app_file: app.py
models:
  - Qwen/Qwen3.5-9B
---

# Music Japan ZeroGPU app — deployment template, not yet deployed

This folder is a **GitHub-sourced Hugging Face Gradio ZeroGPU Space template**. It has *not* been allocated an HF GPU or successfully launched until a verified HF Space URL and runtime evidence exist.

## Target

- Actual **Qwen/Qwen3.5-9B** open-weight model (Apache-2.0) using Hugging Face Transformers.
- Official **ZeroGPU large** slice: up to 48 GB GPU VRAM, not an all-day dedicated GPU.
- Free account allowance: 5 GPU minutes/day; anonymous visitors 2 minutes/day.
- Requires HF account in good standing (email verified, older than 30 days) and explicit Space ZeroGPU selection; free accounts can host up to two qualifying ZeroGPU Spaces.
- GitHub is source; the HF Space is the deployment target. **No paid GPU or model credits may be enabled**.

## How to deploy after owner authorizes HF write access

1. Create a **Gradio** Space on the connected MusicJapanLLC Hugging Face account. Choose free **ZeroGPU** and verify dashboard hardware (not paid T4/L4/A100).
2. Copy this folder's `README.md`, `app.py`, and `requirements.txt` to the root of that Space repo. Model weights are public; no proprietary API credentials needed.
3. Check build logs, runtime health, then send a test prompt and record output and GPU hardware readback.
4. Stop if the platform requests a paid plan, payment card or prepaid GPU credits. A GitHub commit is NOT proof of HF deployment.
5. Source privacy: do not connect company databases, private GitHub repos, browser cookies or user chats to this public Space.

## Verification constraints

No promise of Anthropic Fable/Opus identity or performance. 9B model is open-weight and can be independently audited.

ZeroGPU docs: https://huggingface.co/docs/hub/spaces-zerogpu

Original model: https://huggingface.co/Qwen/Qwen3.5-9B

Space documentation: https://huggingface.co/docs/hub/spaces-overview
