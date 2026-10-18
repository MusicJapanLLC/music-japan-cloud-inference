# OpenRouter zero-cost connector — Music Japan R&D

Source: openrouter_free_client.py (Python 3.10+, standard library only)

- Explicitly allowed: NVIDIA Nemotron 3 Ultra 550B A55B (free); Thinking Machines Inkling 975B A41B (free, **agentic harness use only**); Poolside Laguna S 2.1 (free).
- Not permitted: paid variants, silent subscription/paid API fallback, prompts containing confidential or personal/customer material, key sharing, quota evasion.
- OpenRouter models are hosted by third parties. Invoking them **is not** owning or running their weights.
- Free model endpoints have account-wide limits. Never retry a 401/403/429 automatically, or try another account/IP to evade limits.
- OpenRouter free providers may use prompts/responses to improve their services; submit only owner-approved public synthetic examples.
- The client enforces a live catalog check over all reported pricing dimensions and sets a hard per-provider \`max_price={"prompt":0,"completion":0}\` ceiling.
- A genuine API key is required to actually generate. Obtaining a model catalog (HTTP 200) **does not** establish model inference.

## On Windows (per authorized PC)

1. Visit https://openrouter.ai/keys and sign in with your own account.
2. Create a standard API key with a **$0 spending limit** if supported in the dashboard. Do not create a management key.
3. Keep it only in the Windows user's private profile at \`%LOCALAPPDATA%\MusicJapan\APIForge\secrets\openrouter-free.token\`. On the existing Music Japan notebook the already-installed \`FrontierRuntime\install_openrouter_free_key.ps1\` prompts with hidden input, validates the key, and writes the file with an owner-only ACL. Never put keys into GitHub, console output or public files.
4. Install \`openrouter_free_client.py\` beside existing tools without overwriting APIForge releases or changing existing listeners.

Verify catalog without authentication:

    python openrouter_free_client.py --models

Verify key only, without generating:

    python openrouter_free_client.py --check-key

Execute a small **public, synthetic** generation after key enrollment:

    python openrouter_free_client.py --public --model nvidia/nemotron-3-ultra-550b-a55b:free --prompt "Compute 6*7. Return only the integer." --max-tokens 64 --json

Coding-model example:

    python openrouter_free_client.py --public --model poolside/laguna-s-2.1:free --prompt "Write a Python function to reverse a string." --max-tokens 256 --json

Inkling's free endpoint is restricted to actual agentic harnesses. Use this in an authorized coding/agent framework, and pass \`--agentic-harness\` **only** when that is truly the execution environment.

## Evidence to retain

- Exact requested and upstream-reported model IDs (an alias or local routing name alone does not prove underlying weights)
- Whether the model produced real nonempty text and finished with \`stop\`
- Token usage when supplied by OpenRouter, elapsed time, date, and prompt digest
- Failures (401, 403, 429, quota exhausted, empty or truncated output) as failures, not successes

The client deliberately refuses redirects, unknown model slugs, any nonzero/unknown model price, a missing key, and an unfinished generation. It does not run shell commands requested by models.

Run offline tests:

    python -m unittest -v test_openrouter_free_client.py

### Links

- https://openrouter.ai/docs/api/api-reference/models/get-models
- https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key
- https://openrouter.ai/blog/tutorials/how-to-get-the-lowest-cost-llm-inference-on-openrouter/
- https://openrouter.ai/pricing
- https://openrouter.ai/thinkingmachines/inkling:free
