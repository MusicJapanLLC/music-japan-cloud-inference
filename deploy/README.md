# Music Japan — deployable public reconstructed LLM API

These are genuine public GGUFs with publisher-declared Apache 2.0 weights. They are not the original Anthropic Fable 5.1 or Opus 5.5 models.

## Three downloadable public reconstructions

- `fable-reconstruction-9b`: Empero Qwable-9B, tuned on published Fable 5 / terminal traces
- `fable-reconstruction-27b`: TeichAI Qwen3.6-27B, experimental public Fable-5-derived tuning
- `fable-opus-merge-9b`: public Qwen3.5-9B merged from Fable and Opus 4.6-related distilled checkpoints

Public model file identities and licenses are verified by SHA-256 pinned in `cloud_infer_frontier.py`. Third-party trace-data rights for commercial distribution are NOT independently verified. Internal research is fine, commercial use requires review.

## Run on owner-authorized Linux/cloud machine

1. Install Docker Engine + Compose and Python 3.10+
2. From repo root, run `python3 deploy/prepare_public_model.py --profile fable-reconstruction-9b --download`
3. `cp deploy/.env.example deploy/.env`
4. Generate a random API secret locally using `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` and replace `MJ_INFERENCE_API_KEY` in deploy/.env
5. `cd deploy && docker compose up --pull always`

Only `http://127.0.0.1:18080` is exposed on the host, and the runtime requires the local API key. Never commit .env or weights to GitHub.

The live server supports OpenAI-style `/v1/chat/completions`, `/v1/responses`, `/v1/models`, and Anthropic-shape `/v1/messages` (compatibility is not guaranteed for every upstream feature). It performs real inference from the open-weight model, not a model-provider API.

Example request (replace user-owned LOCAL_KEY):

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Authorization: Bearer LOCAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mj-public-fable-inspired-9b","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

To change model, repeat prepare with `--profile fable-reconstruction-27b` or `--profile fable-opus-merge-9b`, then set `MODEL_FILENAME` and `MODEL_ALIAS` in deploy/.env. 27B needs more RAM and will often be slower than 9B on CPU.

For GPU hosts, the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image supports CUDA, and `--n-gpu-layers` should be adjusted in an owner-controlled deployment. This repo does not automatically provision paid GPUs.

**Deployment status:** This is a complete reusable runtime configuration, but NO persistent 24/7 cloud VM is claimed. A GitHub Actions run is an ephemeral benchmark, not a public production inference endpoint.

Source: https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
