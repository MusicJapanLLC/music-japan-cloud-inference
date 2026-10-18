# Music Japan — pinned 1.6T native runtime and exact-request optimizer

## What this deliverable is

An owner-operated runtime integration for **DeepSeek-V4-Pro-0813**, the published DeepSeek 1.6T-class backbone with its built-in DSpark draft module. This is not a new foundation checkpoint trained by Music Japan, not a bundle of smaller models renamed 1.6T, and not a working allocated GPU cluster.

- Model: `deepseek-ai/DeepSeek-V4-Pro-0813`
- Fixed revision: `72e1d3230f6c080a530b0a1d46f8eb4602340597`
- Publisher safetensors metadata: 1,650,497,936,906 parameter elements, including the draft module
- Official weight manifest: **66 safetensors shards / 892,744,322,880 bytes**
- License: MIT, preserved in `LICENSE.model`
- Canonical source-lock SHA-256: `2ce8f718065196b23a0d444d085133577c0ce3f4e4d852df5ebdd0b72f7294c9`

The actual 892.744 GB weight bytes have **not** been downloaded or executed during this work. Six smaller config/tokenizer/index/license assets totaling 18,022,774 bytes were downloaded from the exact publisher revision and their actual bytes were independently hashed. That is distinct from verifying all model weight bytes.

The current execution policy is **free only**. The paid DeepSeek API workflow and its approval input have been withdrawn, and the `trillion_cloud.py` CLI blocks `--execute` before reading credentials or using the network. The remaining cloud job checks public model metadata only; it does not perform inference. See [CLOUD.md](CLOUD.md). Any free run of a different, smaller model is evidence for that named model only and does not complete the 1650B target.

## Implemented

### Explicit native launch profiles

`trillion_runtime.py` validates the immutable source lock, then requires a real GPU inventory and the selected **vLLM 0.25.0** release before explicit launch. It checks actual sizes and SHA-256 of all 66 weight files plus all six auxiliary files before starting vLLM. Planning itself downloads nothing, creates no paid cloud resources and calls no hosted model API.

Choose a profile explicitly when using the four-GPU path; omitting `--profile` preserves the initial eight-GPU profile. Both use one Linux host, a short 8K/16K/32K context, the same fixed model snapshot and owner authentication.

| Profile | Hardware and parallelism | Free-VRAM planning reserve per GPU |
| --- | --- | --- |
| `cuda-8` (default) | Initial eight distinct Hopper-or-newer CUDA GPUs, TP8 + expert parallelism, four maximum sequences | 134,985,390,256 bytes |
| `gb300-4` | Single 4×GB300 node, SM10.x, TP1 + DP4 + expert parallelism, one maximum sequence per DP rank | 268,897,038,688 bytes |

The four-GPU profile follows the [publisher's recipe at the fixed 0813 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/README.md): DP4 + EP, FP8 KV cache, block size 256, `deep_gemm_mega_moe`, FP4 indexer cache and seven DSpark proposals with greedy draft sampling. TP remains one; this is not TP4 plus DP4, which would require 16 GPUs. Local DP is fixed to four multiprocessing ranks. The selected vLLM 0.25.0 release exposes these configuration fields. The built-in DeepSeek V4 tokenizer remains selected, without enabling arbitrary remote model code. The four-GPU profile uses 0.90 GPU memory utilization and caps sequences at one per DP rank; the default profile keeps 0.85 utilization and probabilistic draft sampling.

Each profile checks its own GPU count, generation and per-device free memory. The SM10.x restriction applies to the GB300 kernel recipe; it does not treat desktop SM12x Blackwell as the same kernel target. GPU names are not used to attest a GB300 node or its interconnect. `topology_verified` remains false in preflight output. The free-VRAM figures are conservative planning heuristics, NOT measured fit guarantees or universal hardware requirements. A real launch can still fail for topology, driver, kernel, replicated dense parameters, memory fragmentation or actual KV usage. Neither profile exposes the model's full million-token context or Think Max mode. CPU offload, AMD and multi-node configurations require separate integration and are not silently selected.

`vllm==0.25.0` is the selected integration pin, not a claim that every newer version is invalid or that this exact launch was GPU-tested. The v0.25.0 source exposes `method='dspark'` and `draft_sample_method='probabilistic'`.

### Optimization without switching to a smaller model

- Native **DSpark**: seven speculative tokens; the original target verifies draft proposals. No synthetic acceptance benchmarking flag is enabled.
- Prefix caching and chunked prefill: reuse common prompt work and bound prefill batches.
- Baseline arm: `--baseline` removes only the DSpark option, supporting a matched comparison when actual hardware is available.
- `SingleFlight`: explicit opt-in, deterministic exact-request coalescing and bounded in-memory cache. Eight simultaneous identical synthetic requests were reduced to one mocked provider call in tests. This is **not eight-times model speed** and is not a native 1.6T benchmark.
- Cache keys include the exact request, fixed source revision and authenticated namespace. Responses have generated/cache/coalesced labels and distinguish source token usage from fresh generation. Stochastic sampling is not silently cached. TTL, bytes, entries, active calls and waiters are bounded.

The optional client is a library and one-shot CLI, not another production listener. Its only endpoint is the owner's authenticated `127.0.0.1:18189`; it never selects a paid model, changes an existing APIForge listener or falls back to 27B/177B/550B.

## Run locally without GPU or model calls

```sh
python -m unittest discover -s trillion -p 'test_*.py' -q
python trillion/trillion_runtime.py --lock trillion/model.lock.json --weights-dir /owned/not-downloaded
python trillion/trillion_runtime.py --profile gb300-4 --weights-dir /owned/not-downloaded
python trillion/trillion_client.py
```

On the current owner notebook this returns a valid plan and explicitly reports zero visible CUDA GPUs / missing selected vLLM. That result is **not a started inference server**. No expensive operation occurs without `--execute`.

## Explicit launch on a separately authorized capable Linux host

Prepare the pinned model files on that host using the normal publisher download tools and a current supported vLLM installation matching this selected integration version. Keep the snapshot immutable while it is verified and served. Do not execute downloaded arbitrary Python source or model plugins. The launch command itself never provisions a machine or downloads the model.

Use an existing owner-only credential file, not a token embedded in the command line or committed to GitHub:

```sh
python trillion/trillion_runtime.py \
  --lock trillion/model.lock.json \
  --profile cuda-8 \
  --weights-dir /owned/models/DeepSeek-V4-Pro-0813 \
  --api-key-file /owned/secrets/trillion-api.key \
  --context 32768 --execute
```

For an already allocated, authorized single 4×GB300 node, select `--profile gb300-4` in that command. The choice changes the hardware and kernel recipe only; all 66 weight hashes, auxiliary assets, vLLM version, explicit `--execute` and owner API-key checks still apply. This profile change has not downloaded the weights, allocated a GPU or performed native inference.

The server binds only loopback. Existing 18186/18187/18188 listeners are untouched. Use the same owner-controlled machine for this initial client; no public tunnel or firewall changes are made:

```sh
python trillion/trillion_client.py --execute --public \
  --api-key-file /owned/secrets/trillion-api.key \
  --prompt-file /owned/public-research/task.txt --max-tokens 4096
```

Only explicitly public technical research inputs are accepted by this initial client. The metadata label is an assertion by the caller, not a secret scanner. Do not send Music Japan private source, customer messages, credentials or private DB contents to any public demo or public CI.

## Verification boundaries

- Four-GPU profile revision: 27 offline runtime-contract tests passed, covering explicit profile selection, per-card VRAM boundaries, wrong GPU count/architecture/version, the publisher's DP4 + EP command and no-launch planning. These are configuration tests with synthetic GPU inventories, not GPU inference or topology measurements
- Local Linux authoring environment: 33 offline tests passed
- Owner Windows notebook: 33 tests, 32 passed and 1 symlink-creation-privilege skip
- Existing repository suite on owner Windows: 155 tests, 153 passed and 2 existing platform skips
- Actual six auxiliary-file hashes passed; full weight-byte verification is implemented but not performed
- One public sponsored **preview-model** canary was attempted through `akhaliq/DeepSeek-V4-Pro`; it returned a provider error event after 1.157s. It is preserved as failure and is NOT an 0813 inference result
- No cloud GPU was allocated; no new company API charge or subscription-model fallback was used
- DSpark latency, throughput, acceptance length and coding quality remain to be measured on actual authorized hardware. Unit tests, a generated command, hashes and model parameter count do not establish those results

`evidence.json` records the scope and immutable identifiers. The source-lock hash is an integrity anchor, not a publisher digital signature. The code has not been security-certified against a compromised host; use an isolated owned machine and keep its source, credentials and snapshot under normal OS access controls.

## Primary references

- Publisher model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
- Exact config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/config.json
- Publisher DSpark: https://github.com/deepseek-ai/DeepSpec
- vLLM selected speculative config: https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/config/speculative.py
- vLLM model recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro
- vLLM 0.25.0 parallel configuration: https://docs.vllm.ai/en/v0.25.0/api/vllm/config/parallel/
- vLLM 0.25.0 kernel configuration: https://docs.vllm.ai/en/v0.25.0/api/vllm/config/kernel/
- vLLM 0.25.0 attention configuration: https://docs.vllm.ai/en/v0.25.0/api/vllm/config/attention/
