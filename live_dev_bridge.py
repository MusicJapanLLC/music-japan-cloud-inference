#!/usr/bin/env python3
"""Real public code repair: existing FrontierRuntime -> secret-free CI Docker checks.

No foundation-model equivalence claim. No generated source is run on the PC.
The GitHub transport is deliberately limited to registered public fixtures.
"""
from __future__ import annotations
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import verified_dev_agent as v

MODEL = 'qwen3.8-27b-public'
REPOSITORY = 'MusicJapanLLC/music-japan-cloud-inference'
TASKS = {'sliding-window-retry-budget': 'samples/sliding-window/task.json'}
REQUEST_SCHEMA = 'mj.public-dev-check-request/1.0'
RESULT_SCHEMA = 'mj.public-dev-check-result/1.0'
MAX_WIRE_BYTES = 256_000


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def decode_response(data: dict, expected_model: str) -> tuple[str, dict]:
    try:
        if not isinstance(data, dict) or data.get('model') != expected_model:
            raise ValueError('MODEL_ID_MISMATCH')
        choices = data['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('AMBIGUOUS_CHOICES')
        choice = choices[0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('MODEL_RESPONSE_NOT_COMPLETE')
        raw = choice['message']['content']
        if not isinstance(raw, str) or not raw.strip() or len(raw.encode('utf-8')) > v.MAX_RESPONSE_BYTES:
            raise ValueError('INVALID_RESPONSE_SIZE')
        text = raw.strip()
        normalization = 'none'
        fence = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', text, re.S)
        if fence:
            text = fence[1].strip()
            normalization = 'outer_json_fence_only'
        parsed = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(parsed, dict):
            raise ValueError('RESPONSE_NOT_OBJECT')
        return text, {'strict_json': normalization == 'none', 'normalization': normalization,
                      'raw_content_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                      'model_identity_evidence': 'gateway_response_only_not_weight_attestation'}
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError('INVALID_MODEL_RESPONSE') from exc


def build_request(task: v.Task, workspace: Path, request_id: str) -> dict:
    if task.task_id not in TASKS or not re.fullmatch('[0-9a-f]{32}', request_id):
        raise ValueError('PUBLIC_TASK_OR_REQUEST_ID_NOT_REGISTERED')
    original, candidate = v.workspace_manifest(task.workspace), v.workspace_manifest(workspace)
    if set(original) != set(candidate):
        raise ValueError('NON_TARGET_WORKSPACE_CHANGE')
    for path in original:
        if path not in task.targets and original[path] != candidate[path]:
            raise ValueError('NON_TARGET_WORKSPACE_CHANGE')
    edits = [{'path': name, 'content': (workspace / name).read_text('utf-8')}
             for name in task.targets if original[name] != candidate[name]]
    if edits:
        edits = v.parse_edits(json.dumps({'edits': edits}), task.targets)
    return v.signed({'schema': REQUEST_SCHEMA, 'request_id': request_id,
                     'data_class': 'public', 'task_id': task.task_id,
                     'baseline_manifest': original, 'issue_sha256': v.sha(task.issue),
                     'edits': edits})


def validate_request(data: dict, root: Path) -> tuple[v.Task, list[dict]]:
    fields = {'schema', 'request_id', 'data_class', 'task_id', 'baseline_manifest',
              'issue_sha256', 'edits', 'sha256'}
    if (not isinstance(data, dict) or set(data) != fields or not v.verified(data)
            or data.get('schema') != REQUEST_SCHEMA or data.get('data_class') != 'public'
            or not isinstance(data.get('request_id'), str)
            or not re.fullmatch('[0-9a-f]{32}', data['request_id'])
            or data.get('task_id') not in TASKS or len(v.canonical(data)) > MAX_WIRE_BYTES):
        raise ValueError('INVALID_PUBLIC_CHECK_REQUEST')
    task = v.load_task(root / TASKS[data['task_id']])
    if data['baseline_manifest'] != v.workspace_manifest(task.workspace) or data['issue_sha256'] != v.sha(task.issue):
        raise ValueError('BASELINE_MANIFEST_MISMATCH')
    if data['edits'] == []:
        return task, []
    return task, v.parse_edits(json.dumps({'edits': data['edits']}), task.targets)


def verify_request(data: dict, root: Path, runner: v.Runner | None = None) -> dict:
    task, edits = validate_request(data, root)
    with tempfile.TemporaryDirectory(prefix='mj-public-check-') as td:
        work = Path(td) / 'workspace'
        shutil.copytree(task.workspace, work)
        if edits:
            v.apply_edits(work, edits)
        check = (runner or v.DockerRunner()).check(work)
    return v.signed({'schema': RESULT_SCHEMA, 'request_id': data['request_id'],
                     'request_sha256': data['sha256'], 'check': dataclasses.asdict(check),
                     'runner': 'docker-network-none-no-secrets',
                     'github_commit': os.environ.get('GITHUB_SHA'),
                     'github_run_id': os.environ.get('GITHUB_RUN_ID'),
                     'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')})


def decode_check(result: dict, request: dict) -> v.Check:
    if (not v.verified(result) or result.get('schema') != RESULT_SCHEMA
            or result.get('request_id') != request['request_id']
            or result.get('request_sha256') != request['sha256']
            or result.get('runner') != 'docker-network-none-no-secrets'):
        raise ValueError('UNTRUSTED_OR_MISMATCHED_CHECK_RECEIPT')
    try:
        value = result['check']
        check = v.Check(**value)
        if (type(check.passed) is not bool or not isinstance(check.diagnostic, str)
                or len(check.diagnostic) > 3500
                or type(check.elapsed_seconds) not in (int, float)
                or not math.isfinite(check.elapsed_seconds) or check.elapsed_seconds < 0
                or (check.exit_code is not None and type(check.exit_code) is not int)
                or (check.tests_run is not None and (type(check.tests_run) is not int or check.tests_run < 0))
                or (check.passed and (check.exit_code != 0 or not check.tests_run))):
            raise ValueError('INVALID_EXECUTION_RESULT')
        return check
    except (KeyError, TypeError) as exc:
        raise ValueError('INVALID_EXECUTION_RESULT') from exc


class FrontierModel:
    """Reuse the owner's installed client; its token stays in the private client process."""
    def __init__(self, client: Path, evidence: Path, captured_response: Path | None = None,
                 captured_prompt: Path | None = None, max_tokens: int = 2048):
        if not client.is_file() or client.name != 'frontier_client.py' or not 128 <= max_tokens <= 4096:
            raise ValueError('EXISTING_CLIENT_OR_BUDGET_INVALID')
        if (captured_response is None) != (captured_prompt is None):
            raise ValueError('REPLAY_REQUIRES_ORIGINAL_PROMPT_AND_RESPONSE')
        self.client, self.evidence, self.max_tokens = client, evidence, max_tokens
        self.replay, self.replay_prompt = captured_response, captured_prompt
        self.calls = []

    def generate(self, model_name: str, prompt: str) -> str:
        if model_name != MODEL:
            raise ValueError('UNAPPROVED_FRONTIER_PROFILE')
        started = time.monotonic()
        replay = self.replay is not None
        if replay:
            if self.replay_prompt.read_text('utf-8-sig') != prompt:
                raise ValueError('CAPTURED_PROMPT_MISMATCH')
            raw = self.replay.read_text('utf-8-sig')
            self.replay = self.replay_prompt = None
        else:
            process = subprocess.run([sys.executable, str(self.client), '--public',
                                      '--max-tokens', str(self.max_tokens), '--timeout', '660',
                                      '--reasoning-effort', 'low'], input=prompt, shell=False,
                                     capture_output=True, encoding='utf-8', errors='replace', timeout=680)
            if process.returncode != 0:
                raise RuntimeError('EXISTING_FRONTIER_CLIENT_FAILED')
            raw = process.stdout
        if len(raw.encode('utf-8')) > MAX_WIRE_BYTES:
            raise ValueError('FRONTIER_RESPONSE_TOO_LARGE')
        response = json.loads(raw, object_pairs_hook=unique_object)
        text, info = decode_response(response, model_name)
        # Keep only public answer fields. Never copy client config, token, env or private DB.
        capture = v.signed({'schema': 'mj.live-code-generation/1.0', 'model': model_name,
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                            'response': {'model': response['model'], 'choices': [{'finish_reason': response['choices'][0]['finish_reason'], 'message': {'role': 'assistant', 'content': response['choices'][0]['message']['content']}}],
                                         'usage': response.get('usage')},
                            'format': info, 'replayed_actual_capture': replay,
                            'call_seconds': round(time.monotonic() - started, 3),
                            'original_inference_seconds': None,
                            'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')})
        self.evidence.mkdir(parents=True, exist_ok=True)
        filename = self.evidence / ('generation-' + str(len(self.calls) + 1) + '.json')
        filename.write_bytes(v.canonical(capture) + b'\n')
        self.calls.append({'capture_sha256': capture['sha256'], 'format': info, 'replayed': replay})
        return text


class GitHubRunner:
    """Bounded cloud test transport using normal git auth, never extracted tokens."""
    evidence_name = 'github-public-ephemeral-docker-network-none'

    def __init__(self, root: Path, task: v.Task, branch: str, public: bool = False, timeout: int = 600):
        if not public or not re.fullmatch(r'runs/vda/[a-z0-9-]{1,60}', branch) or not 60 <= timeout <= 900:
            raise ValueError('PUBLIC_CONSENT_OR_ISOLATED_BRANCH_REQUIRED')
        self.root, self.task, self.branch, self.timeout = root, task, branch, timeout
        self.receipts = []

    def git(self, *arguments, allow_failure=False):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
        done = subprocess.run(['git', *arguments], cwd=self.root, env=env, shell=False,
                              capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        if done.returncode and not allow_failure:
            raise RuntimeError('GIT_OPERATION_FAILED:' + arguments[0])
        return done

    def check(self, workspace: Path) -> v.Check:
        started = time.monotonic()
        if self.git('branch', '--show-current').stdout.strip() != self.branch:
            raise ValueError('WORKTREE_BRANCH_CHANGED')
        expected_origin = 'https://github.com/' + REPOSITORY + '.git'
        if self.git('remote', 'get-url', 'origin').stdout.strip() != expected_origin:
            raise ValueError('UNAPPROVED_PUBLIC_REPOSITORY')
        if self.git('diff', '--name-only', '--cached').stdout.strip():
            raise ValueError('UNRELATED_STAGED_CHANGES')
        request = build_request(self.task, workspace, uuid.uuid4().hex)
        path = 'dev_requests/' + request['request_id'] + '.json'
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(v.canonical(request) + b'\n')
        self.git('add', '--', path)
        self.git('-c', 'user.name=Music Japan Dev Agent', '-c', 'user.email=dev-agent@music-japan.invalid',
                 'commit', '-m', 'test: request isolated public candidate ' + request['request_id'])
        sent_sha = self.git('rev-parse', 'HEAD').stdout.strip()
        self.git('push', 'origin', 'HEAD:refs/heads/' + self.branch)
        last_sha = sent_sha
        print(json.dumps({'phase': 'cloud_check_submitted', 'request_id': request['request_id']}), flush=True)
        while time.monotonic() - started < self.timeout:
            time.sleep(15)
            remote = self.git('ls-remote', '--heads', 'origin', 'refs/heads/' + self.branch).stdout.split()
            if not remote or remote[0] == last_sha:
                continue
            last_sha = remote[0]
            self.git('fetch', 'origin', self.branch)
            result = self.git('show', 'FETCH_HEAD:dev_results/' + request['request_id'] + '.json', allow_failure=True)
            if result.returncode:
                continue
            if len(result.stdout.encode()) > MAX_WIRE_BYTES:
                raise ValueError('CLOUD_RECEIPT_TOO_LARGE')
            record = json.loads(result.stdout, object_pairs_hook=unique_object)
            if record.get('github_commit') != sent_sha:
                raise ValueError('CHECKER_COMMIT_MISMATCH')
            check = decode_check(record, request)
            self.git('merge', '--ff-only', 'FETCH_HEAD')
            self.receipts.append({'request_id': request['request_id'], 'request_sha256': request['sha256'],
                                  'result_sha256': record['sha256'], 'github_run_id': record.get('github_run_id')})
            print(json.dumps({'phase': 'cloud_check_received', 'passed': check.passed, 'tests': check.tests_run}), flush=True)
            return check
        return v.Check(False, None, round(time.monotonic() - started, 3), 'CLOUD_RECEIPT_TIMEOUT', None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help='CI only: validate and execute a public request')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--task-id', choices=tuple(TASKS), default='sliding-window-retry-budget')
    parser.add_argument('--client', type=Path)
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--branch', default='runs/vda/live-20261018')
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--captured-response', type=Path)
    parser.add_argument('--captured-prompt', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evidence/live-dev-20261018'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.worker:
        if not args.result or args.worker.stat().st_size > MAX_WIRE_BYTES:
            raise ValueError('BOUNDED_WORKER_FILES_REQUIRED')
        request = json.loads(args.worker.read_text('utf-8'), object_pairs_hook=unique_object)
        record = verify_request(request, root)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_bytes(v.canonical(record) + b'\n')
        print(json.dumps({'request_id': record['request_id'], 'check': record['check']}))
        return 0  # A failing assertion is a valid measured result, not a workflow infrastructure failure.
    if not args.execute:
        print(json.dumps({'mode': 'dry_run', 'task_id': args.task_id, 'model': MODEL,
                          'cloud': REPOSITORY, 'generations': 0, 'checks_dispatched': 0}))
        return 0
    if not args.public or not args.client or not 1 <= args.attempts <= 3:
        raise ValueError('EXPLICIT_PUBLIC_EXECUTION_CLIENT_AND_BUDGET_REQUIRED')
    task = v.load_task(root / TASKS[args.task_id])
    runner = GitHubRunner(root, task, args.branch, public=True)
    model = FrontierModel(args.client, args.output, args.captured_response, args.captured_prompt)
    record, edits = v.execute(task, model, runner, [MODEL], args.attempts)
    record = v.signed({**record, 'live_model_calls': model.calls, 'cloud_checks': runner.receipts,
                       'upstream_billing_provenance': 'existing_public_only_frontier_client_no_paid_fallback',
                       'foundation_weights_created': False})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'receipt.json').write_bytes(v.canonical(record) + b'\n')
    if edits is not None:
        (args.output / 'selected-edits.json').write_bytes(v.canonical(edits) + b'\n')
    print(json.dumps({'status': record['status'], 'model': MODEL, 'cloud_checks': len(runner.receipts),
                      'attempts': len(model.calls), 'receipt_sha256': record['sha256']}))
    return 0 if record['status'] == 'PASS_VERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
