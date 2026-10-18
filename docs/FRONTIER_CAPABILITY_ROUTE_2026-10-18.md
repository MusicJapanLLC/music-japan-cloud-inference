# Music Japan — Frontier Capability Route / 2026-10-18

## Goal and hard evidence rule

The target is successful **software-engineering work**, not a parameter-count contest.
A model must not be called equivalent to GPT-6 Astra, Claude Fable 5.1, or Gemini 4 Argon until the **same task distribution, agent harness, time budget, and hardware cost** have been compared in independent reproducible evaluations.

The current `cloud_infer_frontier.py` suite has 12 simple synthetic tasks. It checks weight loading and basic correctness, but not repository-scale coding autonomy.

## Nonstandard route: counterexample-guided code synthesis

`verified_dev_agent.py` implements the following:

1. Read a user-owned, intentionally bounded source repository and a reproducible issue.
2. Request independent repairs from authorized localhost OpenAI-compatible model profiles, only with listed source files in the prompt. No hidden tests are supplied.
3. Parse strict JSON *complete file replacements*, denying unapproved paths, traversal, symlinks, and oversize output.
4. Copy every candidate into its own temporary workspace.
5. Run the test suite in a Docker container with `--network none`, read-only filesystem, dropped capabilities, non-root user, CPU/RAM/PID limits, and **no host credentials mounted**. The Docker image must already exist locally.
6. Feed only bounded, untrusted failure diagnostics into the next iteration. Model assertions of success do not count.
7. Export verified edits separately from a content-free SHA-256 integrity receipt. Do not publish internal code.
8. Preserve failures as failures. A passing result requires the test suite to have actually run; test counts must agree when known.

This approach improves **task completion odds** through independent tests and automated repair. It **does not** change the model's weights, guarantee general intelligence, or authenticate the model behind a localhost proxy.

`test_verified_dev_agent.py`: 15 offline unit tests passed on 2026-10-18. This confirms control-flow guards, not real model quality. The bundled `samples/sliding-window` contains an intentionally defective thread-safety/rolling-window implementation with executable acceptance tests.

## How to execute on owned / permitted infrastructure

Prerequisites: Python 3.11+, a real owner-authorized local OpenAI-compatible inference server bound to loopback, Docker daemon and pre-pulled `python:3.12-alpine` image, CPU/RAM/disk capacity sufficient for that chosen model and container. Work in a clean, public-data-only GitHub runner or a dedicated isolated PC. **No raw ChatGPT messages, LINE data, credentials, or customer records should enter a public CI runner.**

```bash
python3 -m unittest -q test_verified_dev_agent
python3 verified_dev_agent.py \
  --task samples/sliding-window/task.json \
  --models locally-available-model-id --dry-run
python3 verified_dev_agent.py \
  --task samples/sliding-window/task.json \
  --models locally-available-model-id \
  --attempts 3 \
  --port 18184 \
  --receipt out/receipt.json \
  --selected out/selected-edits.json
```

The test-runner uses Docker **without automatically pulling an image**. Never place sensitive directories inside a test fixture. A Docker container reduces risk but is not a formal sandbox guarantee against all kernel/runtime vulnerabilities; use an isolated disposable worker for untrusted code.

## Legitimate paths to genuinely own model capabilities

- **Owned model weights**: obtain an explicitly permitted base model (for example, Qwen3.5-397B-A17B is listed Apache-2.0 by its official Hugging Face model card), preserve its license/provenance, and train original LoRA/full fine-tunes on rights-cleared training data. File availability alone is not an endorsement of an ultra-low-bit quantization or its correctness.
- **Owned data and tooling**: derive tests, input-output pairs, code repairs, and regression suites from Music Japan's own source, consented contributions, permissively licensed datasets, or a written license. Retain exact origin, scope of rights and allowed uses. Competing-model output collection/distillation is only eligible when the specific provider terms and written authorizations permit it.
- **Authorized black-box protocol research**: research published/contractually accessible interfaces, error handling, schemas, latency and reliability with permission. This can yield interoperable tools but does not reveal or transfer a provider's hidden weights. Do not evade access controls, rate limits, or confidentiality.
- **Compute acquisition**: research institutional GPU access, grants, sponsorship, time-limited loaned infrastructure, and already-owned machines **without claiming allocations before a real entitlement is granted**. Separate memory, disk and GPU requirement checks. Aggregate CPU runner vRAM is not the same as a unified GPU or 64 GiB RAM host.

## Verified hardware / feasibility, 2026-10-18

- Desktop `DESKTOP-1S0IL8F`: Core i5-9500, ~8 GB RAM, Intel UHD 630, ~2.7 GiB free disk.
- Notebook `My-Note-PC`: Core i5-1145G7, ~8 GB RAM, Intel Iris Xe, ~15.2 GiB free disk.
- GitHub standard hosted Linux runner (public repository): **4 vCPU, 16 GB RAM, 14 GB SSD** according to the GitHub-hosted runners reference. A free standard runner is neither a dedicated GPU nor a 2 TB storage host.
- Do not claim a 397B model's quality or runtime without a completed, independent, multi-task benchmark with actual GGUF checksum, model identity, inference logs, and task pass/fail metrics.

## Next gates for a frontier-grade result

- Real localhost-model run of the code-repair fixture; persist signed receipt and selected edit; verify tests remain passing outside the optimization loop.
- Multi-task issue benchmark across JS/Python/TypeScript/GitHub Actions with newly authored regression tests and cross-model same-budget comparisons. Do not leak solutions into the prompt.
- External benchmark: use Terminal-Bench 4 and SWE-bench Pro *with their official harnesses and comparable resources*, and report confidence intervals and costs. SWE-bench Verified is no longer a sufficient frontier indicator because of contamination concerns.
- Only **after a demonstrated uplift** add gated production operations, true model finetuning, and machine allocation.

## Primary references

- Qwen official weights: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- GitHub runners: https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- Terminal-Bench: https://github.com/harbor-framework/terminal-bench
- SWE-bench: https://github.com/SWE-bench/SWE-bench
- SWE-bench Verified limitations: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- OpenAI business agreement restrictions: https://cdn.openai.com/osa/openai-services-agreement.pdf