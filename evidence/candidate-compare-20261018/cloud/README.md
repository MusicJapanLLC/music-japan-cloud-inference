# Cloud execution evidence

The four files beside this README were downloaded from the completed GitHub
Actions artifact and preserved without rewriting their contents.

- Run: [39115525493](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39115525493)
- Job: [117640430741](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39115525493/job/117640430741)
- Executed commit: `49b4e94eea440ca076112e29315921ed5e82d9ea`
- Receipt time: `2026-10-18T09:29:35+00:00`
- Original artifact: `frozen-candidate-comparison`, ID `12131393975`
- Downloaded ZIP SHA-256: `00d8a0d22382f8e9a6c529d22f3db104dbd47e0fe39f8c919465dc1421dbd519`
- ZIP digest matched both the GitHub artifact metadata and the upload log

## Observed result

| Stage | Result |
|---|---|
| Existing VDA and new controller tests | 49 passed |
| Intentionally defective baseline | 21 tests executed; failed, exit 1 |
| Captured Qwen27 candidate | 21 tests executed; passed, exit 0 |
| Qwen27 fresh-workspace recheck | 21 tests executed; passed, exit 0 |
| Flash Next capture | Generation failed: `GATEWAY_TIMEOUT_OR_NETWORK` |
| DeepSeek capture | Generation failed: `GATEWAY_HTTP_429` |
| Selected task artifact | `qwen27-1`, `PASS_ACCEPTANCE_RECHECK` |

All three live requests preceded cloud testing. The comparator replayed their
frozen responses; it made zero model API calls in CI. Live request timing remains
in `../captures.metadata.json`, and must not be inferred from the microsecond
replay timings in `comparison.json`.

## Files and integrity

| File | SHA-256 of file bytes |
|---|---|
| `comparison.json` | `2f87a81b0a48bb98f50222a845b54d7ec22b8af12244fa811d667335ee9901ec` |
| `selected-edits.json` | `d75ac26d40ab12e8a9df440646f2d1d334ce752af1353543c35d27e38d6bdbe0` |
| `cloud-context.json` | `83192fc6d58c29162ec0f4eaced53137620a0a7ae48712bc3cf3e0b7de84f84f` |
| `docker-image-id.txt` | `c2b253e09afa8c9035885548e68bb10110232755bb73f484a48f79d511e86d95` |

The receipt's internal canonical-payload SHA-256 is
`5c02b80002e4541240ef9aaf75f42b05878e5a895ddb81e8a5598321cd8687c3`.
It is an integrity checksum, not a cryptographic signer attestation. The selected
edit references that same receipt and task hash.

The baseline, candidate and recheck used the existing VDA Docker runner with
network disabled. The runner label was `ubuntu-24.04`; the actual pulled Python
container image ID is recorded beside the receipt.

This supports a passing repair for one public synthetic lease-completion task.
The acceptance tests run in the same interpreter as candidate code and are
readable inside the container; this is not a protected hidden benchmark. It does
not establish general model superiority, foundation-weight training, independent
weight identity, or new GPU ownership.
