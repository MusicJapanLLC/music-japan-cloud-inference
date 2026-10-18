# Comparing generated code with executable evidence

The existing VDA repairs a task sequentially. This extension accepts up to three
model profiles, freezes their proposals, and checks every proposal against the
same public acceptance suite. It reuses the existing VDA workspace handling,
edit allowlist and network-disabled Docker runner. It does not add a model API.

## What was added

- `compare_candidates.py`: strict model/terminal-response checks, bounded
  candidate generation, independent workspaces, full-suite manifest binding,
  acceptance checks and fresh-copy rechecks
- `gateway_candidate_capture.py`: reuse the operator's existing, SHA-pinned
  gateway engine to capture explicitly public synthetic tasks; generated source
  is never executed on the capture controller
- `samples/lease-finalization/`: a synthetic completion-receipt repair task with
  21 public acceptance tests covering stale workers, epochs, retries, cancellation,
  deadline boundaries, error precedence and input immutability
- `.github/workflows/candidate-compare.yml`: run controller tests and candidate
  checks on a standard Ubuntu GitHub runner, record container image identity,
  and upload the comparison receipt and selected edit
- `frontier_candidates.json`: separately sourced research candidates up to 2.8T;
  it is a research inventory and does not enable those models

## Deterministic selection and evidence

Each configured profile receives the same issue and source. Test source and test
diagnostics are omitted from generation prompts. All candidates are captured
before any candidate receives an acceptance result. A successful candidate must
have a completed final response from the expected model, valid allowlisted edits,
the original test count, a passing acceptance result, and a passing independent
rerun from a new workspace. Selection follows declared profile order then attempt
number among passing candidates. Completion speed and parameter count do not
silently select a winner.

`task_manifest_sha256` binds all source and test bytes to the capture. A changed
test file is rejected before even running the baseline. Receipts retain failures,
source/output hashes and exact selection scope. Existing receipt/selected paths
are rejected so a previous successful artifact cannot stand in for this run.

## Commands

Dry-run, without model or Docker calls:

```sh
python -B compare_candidates.py --task samples/lease-finalization/task.json
```

Capture with an existing authorized gateway installation. Supply the installed
engine and local gateway credential paths; the credential remains in the local
transport and is not written into the capture or sent to GitHub.

```sh
python -B gateway_candidate_capture.py \
  --task samples/lease-finalization/task.json \
  --engine /path/to/reviewed/parallel_engine.py \
  --token-file /path/to/local/gateway.token \
  --host 127.0.0.1 --routes qwen27 flashnext deepseek \
  --max-tokens 4096 --timeout 240 \
  --output /path/to/new/captures.json --public --execute
```

The engine must match `ENGINE_SHA256`; inspect a changed version before updating
the pin. Capture produces a strict replay file and a separate `.metadata.json`
file containing observed generation timings. The replay consumer labels its work
as replay, not a new live model request.

On the trusted cloud controller, with the existing Docker image pre-pulled:

```sh
python -B compare_candidates.py \
  --task samples/lease-finalization/task.json \
  --captures evidence/candidate-compare-20261018/captures.json \
  --execute --concurrency 3 --budget-seconds 450 \
  --receipt out/comparison.json --selected out/selected-edits.json
```

## Fresh capture on 2026-10-18

The three requests used the same 3,551-character public issue/source prompt,
4,096-token output cap, low effort and 240-second transport timeout. The returned
capture is preserved in `evidence/candidate-compare-20261018/`.

| Gateway profile | Observed result | Observed request duration |
|---|---|---:|
| `qwen27` | Complete final code response, `stop` | 164.516 s |
| `flashnext` | `GATEWAY_TIMEOUT_OR_NETWORK` | 240.031 s |
| `deepseek` | `GATEWAY_HTTP_429` | 0.031 s |

The successful response reported `qwen3.8-27b-public` and a Qwen3.8-27B Q8 GGUF
upstream identity. This is gateway/upstream-reported identity, not independent
weight-byte attestation. Failed captures remain part of the comparison input.
The original capture predates the immutable-copy improvement. Its exact transfer
snapshot and final manifest were matched independently; the later improvement is
not retroactively attributed to that capture.

The intentionally defective baseline fails 14 of the 21 acceptance tests when
checked against a trusted reference during fixture construction. Actual generated
candidate acceptance and recheck results are reported by the cloud workflow,
not inferred from source inspection or the controller unit tests.

Controller verification: `python -B -m unittest -q test_compare_candidates
test_gateway_candidate_capture test_verified_dev_agent` passed 49 tests at this
implementation snapshot. These offline tests are distinct from the 21 tests of
the generated code.

## Limits of the result

The current VDA runner executes public acceptance tests and candidate code in the
same Python interpreter inside the container. The tests are readable there.
This is not a protected hidden benchmark or a defense against all candidate-side
test-runner interference. The receipt selects a task artifact; it does not certify
a model's general quality or equivalence to any frontier proprietary model.

Timeout-aware adapters remain responsible for stopping their own requests. Python
threads cannot forcibly terminate arbitrary adapter code; late responses are
rejected. This change neither trains foundation weights nor provisions a GPU.
The research inventory documents separate licensing, endpoint and hardware
requirements for larger candidates.
