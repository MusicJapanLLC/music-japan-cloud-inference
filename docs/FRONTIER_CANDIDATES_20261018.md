# Cloud frontier model candidates — 2026-10-18

Public primary-source research only  
Compiled at 2026-10-18T08:58:31Z

## What this inventory means

`frontier_candidates.json` records one Nemotron baseline and five candidates  
Every entry has `executable: false`: no inference, account entitlement, GPU allocation, or deployment was measured for this inventory

A provider marked live is metadata, not successful inference  
An API connection grants access to a service; it does not transfer ownership of the underlying weights  
Downloading licensed weights, hosting them, and training a derivative are separate milestones  
Parameter counts do not establish quality, and this document makes no claim of superiority to GPT-6 Astra, Fable 5.1, or Gemini 4 Argon

## Candidate map

| Model | Parameter evidence | Route observed | Hardware evidence |
|---|---|---|---|
| [Nemotron 3 Ultra NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/raw/main/README.md) | Developer: 550B total / 55B active | [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/nvidia/nemotron-3-ultra-550b-a55b:free/endpoints) lists Nvidia at zero token price; returned quantization is unknown | Official card: 4× B200/GB200/GB300/B300 or 8× H100; not run here |
| [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 552B backbone + 196B Engram + other modules; full checkpoint is not 552B | [Official API mapping](https://api-docs.deepseek.com/quick_start/pricing/): `deepseek-flash` → 1017; paid inference | Base-release reference uses TP8 conversion; no verified 1017 sizing |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Developer: 2.8T total / 104B active | HF metadata lists Together, Fireworks, Featherless, Baseten, DeepInfra | [vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3): minimum 8× GB300 |
| [Qwen 3.8 2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | Developer: 2.4T total / 95B active | HF metadata lists Novita, Together, Fireworks, DeepInfra | [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B): NVFP4 weights 1.32 TiB, 8× B300 or 2 GB300 trays; FP8 2.27 TiB |
| [GLM 5.3](https://huggingface.co/zai-org/GLM-5.3) | HF rounded tensor count approximately 753B; active count not independently verified here | HF metadata lists seven providers | Not independently sized in this inventory |
| [GLM 5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Developer: 320B total / 18B active | [OpenRouter free catalog](https://openrouter.ai/z-ai/glm-5.3-flash:free) exists; operational endpoint was not confirmed | [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash): FP8 weights ~306 GiB before overhead, TP4 on GB200 |

Provider prices and availability can change  
A successful account-specific preflight is required before use  
The JSON distinguishes catalog pages, provider metadata, official API documentation, and missing/deprecated routes

## Corrections that affect implementation

### Model and API variants

- DeepSeek V4.1's [official README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/README.md) distinguishes backbone, Engram memory, and other modules; its 8B prefill / 16B decode activity is not a storage requirement
- The 1017 update is a separate checkpoint; the available [base inference example](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/README.md) is not evidence of a tested 1017 deployment
- Qwen's open checkpoint is text-only and always-thinking; official Qwen3.8-Max API adds features such as vision and non-thinking
- Kimi requires complete assistant messages, including `reasoning_content` and `tool_calls`, in subsequent turns
- GLM 5.3 and Flash document `reasoning_effort` values `low/high/max`; adapter defaults should follow the actual model and workload
- Read advertised evaluation scores as results under the publisher's task, harness, and budget; compare candidate routes with a common application test before promotion

### Licenses

| Artifact | Current license evidence |
|---|---|
| Nemotron Ultra NVFP4 | Current card identifies [OpenMDW-1.1](https://openmdw.ai/license/1-1/) as governing download terms; full license text was not reviewed here |
| DeepSeek V4.1 Flash / 1017 | Official model repositories identify MIT |
| Kimi K3 | [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE); permits modification and deployment with notices and stated business-scale conditions |
| Qwen 3.8 2.4T | [Qwen3.8-Max License](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/LICENSE); permits modification and deployment with notices and stated business-scale conditions |
| GLM 5.3 | [GLM-5.3 License](https://huggingface.co/zai-org/GLM-5.3/raw/main/LICENSE), not the MIT license of GLM 5.2 |
| GLM 5.3 Flash | [MIT](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/LICENSE) |

Model licenses and a hosted API's service terms are different documents  
This inventory does not infer permission to reuse API outputs for training from a downloadable-weight license

### Stale free routes

The [GLM 5.2 free endpoint lookup](https://openrouter.ai/api/v1/models/z-ai/glm-5.2:free/endpoints) returned an empty endpoint list despite a free model page  
The [NVIDIA Kimi K3 page](https://build.nvidia.com/moonshotai/kimi-k3) labels its Free Endpoint deprecated and Partner Endpoint available  
GLM 5.3 Flash's free catalog is therefore a candidate for a fresh endpoint check, not a ready route

## Compute without a new purchase

[ZeroGPU's current documentation](https://huggingface.co/docs/hub/en/spaces-zerogpu) lists 48 GB `large` at 1× quota and 96 GB `xlarge` at 2×  
Included daily quotas are 2 minutes unauthenticated, 5 minutes free, 40 minutes PRO/Team, and 60 minutes Enterprise  
Quota resets 24 hours after first usage  
Free personal accounts in good standing can host up to two Spaces, subject to verified email and account age over 30 days  
Eligible paid tiers can automatically use prepaid credits beyond quota; a zero-new-spend workflow must check allowance and avoid that transition

[HF Inference Providers pricing](https://huggingface.co/docs/inference-providers/en/pricing) currently gives free accounts no monthly credits  
PRO includes USD 2 per month of general compute credits, with Team/Enterprise USD 2 per seat  
An unverified account balance is not free inference

[OpenRouter limits](https://openrouter.ai/docs/api_reference/limits) expose `free_model_daily_requests` through `GET /api/v1/key`  
Use the live account counter rather than legacy hardcoded daily limits  
No account limits were read for this research

### Larger machine grants

| Program | Documented route | Current state |
|---|---|---|
| [HF Community GPU Grant](https://huggingface.co/docs/hub/en/spaces-gpus) | Apply from Space hardware settings using a demonstrable project | No application; no allocation |
| [NVIDIA Inception](https://www.nvidia.com/en-us/startups/) | Free membership and eligible NVIDIA/partner cloud-credit offers | No application; eligibility and award unverified |
| [Google TPU Research Cloud](https://sites.research.google/trc/about/) | Free TPUs for accepted research with expected public results | No application; no allocation |

These are application routes, not already-owned hardware  
A portable demo, reproducible measurements, and a precise resource request make the next step reviewable

## Promotion sequence

1. Confirm current provider endpoints or allocated hardware and checkpoint identity
2. Verify authorized account allowance and permitted request cost
3. Run a small synthetic request and record the final output and actual returned model
4. Validate a tool-call round-trip and preserved conversation fields
5. Run the same application-specific tasks under the same harness and resource limits
6. Promote only measured routes, retaining separate evidence for API inference, self-hosting, and derivative training

No paid runs, new API client, grant applications, or deployments are included in this change
