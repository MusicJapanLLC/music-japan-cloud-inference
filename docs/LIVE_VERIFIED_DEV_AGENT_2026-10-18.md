# Music Japan — public-fixture verified development agent

## Verified result (2026-10-18, JST)

A *real*, prior Qwen3.8-27B gateway response was replayed through the new controller, then evaluated three times on actual GitHub-hosted isolated Docker runners:

| Stage | GitHub run | Result |
|---|---|---|
| Broken baseline | [39108580890](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39108580890) | Original four acceptance tests failed |
| Real model candidate | [39108644070](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39108644070) | Four of four tests passed |
| Fresh independent candidate recheck | [39108691448](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39108691448) | Four of four tests passed |

Original capture: returned model identifier `qwen3.8-27b-public`, `finish_reason=stop`, 1,273 completion tokens. Its response had an **outer Markdown JSON fence**; the controller recorded `strict_json=false` and `normalization=outer_json_fence_only`. The model revision and exact upstream weights are *not* cryptographically attested by an API response.

Completed repair receipt: `sha256=459d7bc022611f93dc71725d54335e5b3124d156802d9dc8a259349501be3816`, `status=PASS_VERIFIED`, `selected_profile=qwen3.8-27b-public`, `cloud_checks=3`, `independent_recheck.tests_run=4`. Local raw model capture, request prompt and selected full source stay on the authorized PC under `%LOCALAPPDATA%\MusicJapan\VerifiedDevAgent-20261018\repo\evidence\live-dev-20261018\`; they have not been published in this branch.

CI for source and controller: [Linux + Windows validation](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39108883105) and [legacy agent verification](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39108883087) succeeded. A notebook suite ran 139 tests with zero failures and two platform-specific skips.

## Re-run with existing authorized infrastructure

Use **only explicit public fixtures**. A GitHub-hosted runner is an ephemeral test machine, not a provisioned LLM GPU. The inference route is the separately configured owner-approved APIForge FrontierRuntime gateway. This workflow never starts or edits a hosted production deployment, does not load private company datasets, and does not request paid model credentials.

On an already-authenticated owner machine with `frontier_client.py` installed and healthy:

```powershell
git clone --config core.autocrlf=false https://github.com/MusicJapanLLC/music-japan-cloud-inference.git mj-public-dev
cd mj-public-dev
git switch -c runs/vda/recheck-20261018
python -m unittest discover -p 'test_*.py' -q
python live_dev_bridge.py --public --execute --task-id sliding-window-retry-budget `
    --client "$env:LOCALAPPDATA\MusicJapan\FrontierRuntime\frontier_client.py" `
    --branch runs/vda/recheck-20261018 --attempts 2
```

The runner checks that its branch is `runs/vda/*`, the remote is precisely this public repository, and only allowlisted fixture source has changed. It stages and pushes bounded request JSON, waits for the CI feedback commit, verifies request SHA and commit linkage, then runs a second, fresh candidate check. It fails closed on timeouts, invalid model IDs, unbounded responses, missing test counts, invalid paths, and missing Docker execution evidence.

**Rules:** No customer messages, Music Japan private code, private DB, credentials or proprietary model outputs in public runners or prompts. This is verified **agent-workflow repair capability**; it is not a new 120B+ foundation model, new GPU allocation or proof of GPT-6/Astra/other closed-model parity.

The receipt `sha256` is a reproducible *integrity digest*, not a key-signed cryptographic identity proof.

## Subsequent fresh-inference verification

The preceding section records the earlier replay-only checkpoint. A subsequent execution called the existing FrontierRuntime client after the failing baseline, without replaying a saved answer, and passed both the candidate and fresh final cloud check. See [fresh generation and full receipt](../evidence/live-dev-fresh-20261018/receipt.json) and [Japanese operating notes](LIVE_DEVELOPMENT_AGENT_2026-10-18.md). Sanitized final-answer captures and selected source are now published in `evidence/`; raw provider reasoning and local credentials remain excluded.

On Windows, use the documented clone configuration to preserve the public fixture bytes. This was checked with a fresh local clone against the manifest from the actual Linux cloud baseline; do not reset an unrelated worktree to normalize its files.
