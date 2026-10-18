# Live Verified Development Agent Implementation Plan

> For agentic workers: use executing-plans for this owner-approved continuation

**Goal:** Real generation through the existing public-only FrontierRuntime, isolated executable checks, failure feedback, repair, and fresh final verification
**Architecture:** Retain verified_dev_agent.py. Add a registered-public-fixture GitHub Docker runner and existing-client adapter. Never run generated source on a personal PC
**Tech stack:** Python 3.12 standard library, existing git authentication, GitHub Ubuntu, Docker
**Spec:** docs/FRONTIER_CAPABILITY_ROUTE_2026-10-18.md

## Constraints
No paid or subscription model calls; no production runtime modifications; no private company data in public CI; no credentials in prompts or evidence; no changes to Dots' ongoing GPU work

## Review focus
Missing test counts, incomplete/model-substituted responses, immutable request/receipt linkage, unchanged acceptance tests, and fresh final verification

## Task 1: repair controller
- [x] Reproduce missing test-count guard, missing previous-source feedback, unbound test manifest, missing final recheck and timeout cleanup
- [x] Correct minimal implementation and run offline suite
- [x] Make Windows symlink privilege limitation an explicit skip; keep full Linux coverage

## Task 2: live transport
- [x] Red/green response-format, candidate-only upload, request digest, and result validation tests
- [x] Implement existing-client reuse and cloud Docker verification with a separate feedback writer
- [x] Exclude upstream reasoning traces from public evidence; explicit outer-fence normalization

## Task 3: live acceptance
- [ ] Run original failing baseline, actual captured model output and fresh candidate recheck in real Docker
- [ ] Record honest generation provenance, Windows/Linux tests and cloud run IDs
- [ ] Review diff, verify final CI, integrate implementation and evidence

## Ledger
The preceding PR #6 is already merged at cfa8fb060ed1da18b7e7191cc93a53df2f7ae5d3
Notebook baseline: 116 tests; WinError 1314 in symlink test
Current offline suite: 139 tests, two platform-specific skips on non-admin Windows; no claimed live-model success from these tests
Captured real public Qwen3.8-27B response: 1273 completion tokens, stop termination, one 1215-character source replacement; strict JSON failed solely due to an outer Markdown fence
