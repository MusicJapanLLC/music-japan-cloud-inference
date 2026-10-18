# Music Japan Adaptive Supervisor

**Own goal:** automatically manage reasoning budget and genuine open-weight models, with correct answer provenance and zero new provider API credits.

## Actual owner-facing workflow (free ephemeral GitHub Linux CPU)

[Actions / Music Japan Adaptive Supervisor](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/workflows/adaptive-supervisor.yml)

Choose a PUBLIC synthetic/research question, run mode `auto` or `force-ultra`, and turn on `execute`. The workflow retrieves real SHA-pinned publicly licensed models; the user never buys an Anthropic/OpenAI hosted API token.

The workflow stages are:

1. **Plan:** deterministic question grounded classification and budget cap. Authentic pinned `gpt-oss-20b` initial model, optionally OpenAI publicly released `gpt-oss-120b`. Model roles are distinct and never masquerade as GPT-6 or closed Claude models.
2. **Draft:** at least two high-reasoning actual gpt-oss20B rounds via llama.cpp on a Github-hosted CPU.
3. **Control:** verify SHA and genuine model source, then independently verify calculations/date or curated translation. In `auto`, only exit early if **all** raw 20B rounds independently verify and no round hits generation budget; otherwise spend one or two more 120B high-effort passes.
4. **Final 120B (conditional):** when escalated, use original OpenAI gpt-oss120B MXFP4 SHA-pinned GGUF on an ephemeral cloud CPU; verify this genuine model, its thinking preset and its cryptographic parent references to the draft.
5. **Selection:** if 120B checks independently correct, prefer it; if 120B is wrong and 20B is independently correct, keep 20B. Unsupported factual claims remain labeled *unverified*, never certified through repeated self-agreement.
6. **Delivery:** answer appears in Github Actions Job Summary and in a technical-only evidence artifact. All models and external inference charges are distinguished.

### Execution receipts

- [18-test original CI and authenticated proposal proof](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39099660141)
- [Actual conditional inference live run](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39099712184). Model gpt-oss20B actually generated two responses in 19.14 and 28.28 seconds. Independent verification succeeded, so **120B was not launched**, as designed. Final artifact SHA-256: `a974c994b2e7ac6445c674af926c1cc81a62ef8ecf42ab77bae2ff1323085dc2`.
- [First real gpt-oss-120b original public GGUF execution](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39095219646). Model original weights 63,387,346,208 bytes; SHA-256 `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`; model loaded and returned a real HTTP 200 answer.
- [Already-running actual two-model MAX/ULTRA](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39098211563). Final acceptance is based on completed signed output, not merely a started GPU or Linux process.

### Constraints & limits

- **No private user data.** Dispatch inputs and GitHub runner logs are on a PUBLIC GitHub repository. Do not send Music Japan customer/LINE/Gmail/secret data. Use the existing private Supabase signed-device gateway for future controlled business retrieval.
- **No extra purchases, new compute charges or model-provider inference tokens.** Standard public GitHub-hosted CPU Actions is used. The provider's free tier and policies can change and must be monitored.
- **Performance:** The tested open gpt-oss120B is *not* GPT-6/Astra, and model council/longer thinking does not prove dominance over proprietary systems. Proof includes actual runtimes, origin checks, and answer acceptance without unsupported claims.
- **No permanent serving host.** GitHub runners are ephemeral and intended for research runs. The output is an actual answer artifact per dispatch.
- **Strongest operating mode:** `force-ultra` always uses original 120B high-effort model after authentic 20B multi-round proposal. `auto` conserves runtime when independent verification succeeds.

## Files

- `adaptive_supervisor.py`: runtime policy, provenance validation, final model selection and summary.
- `test_adaptive_supervisor.py`: adversarial tests including spoofing, wrong model, broken SHA parents, truncated 20B outputs and misleading verification.
- `.github/workflows/adaptive-supervisor.yml`: real GitHub Linux pipeline with conditionally allocated **original** open 120B model.
