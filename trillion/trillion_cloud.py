#!/usr/bin/env python3
"""Public model metadata audit; paid inference is disabled by the free-only policy."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import http.client
import json
import os
import re
import signal
import threading
import time

from trillion_runtime import MODEL, REVISION, canonical, loads

API_HOST = 'api.deepseek.com'
API_PATH = '/chat/completions'
API_MODEL = 'deepseek-v4-pro'
VERSION = 'DeepSeek-V4-Pro-0813'
DOC_HOST = 'api-docs.deepseek.com'
DOC_PATH = '/quick_start/pricing/'
SOURCE_URL = 'https://' + DOC_HOST + DOC_PATH
INPUT_RATE = Decimal('1.32')  # USD / million tokens, reviewed peak cache-miss ceiling.
OUTPUT_RATE = Decimal('3.96')
MAX_RESPONSE_BYTES = 512_000


class CloudError(RuntimeError):
    """Only application-authored error codes; never raw provider/transport errors."""


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def cost(input_tokens, output_tokens):
    return (Decimal(input_tokens) * INPUT_RATE + Decimal(output_tokens) * OUTPUT_RATE) / 1_000_000


class Tables(HTMLParser):
    """Expand HTML row/column spans before selecting a model's price column."""
    def __init__(self):
        super().__init__(); self.tables = []; self.grid = None; self.row = -1; self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            if self.grid is not None:
                raise CloudError('UNSUPPORTED_PRICING_TABLE')
            self.grid = {}; self.row = -1
        elif tag == 'tr' and self.grid is not None:
            self.row += 1
            if self.row > 100:
                raise CloudError('PRICING_TABLE_LIMIT')
        elif tag in ('td', 'th') and self.grid is not None:
            attrs = dict(attrs)
            try:
                rows = int(attrs.get('rowspan', '1')); cols = int(attrs.get('colspan', '1'))
            except (TypeError, ValueError):
                raise CloudError('INVALID_PRICING_SPAN') from None
            if not 1 <= rows <= 20 or not 1 <= cols <= 20 or self.row < 0:
                raise CloudError('INVALID_PRICING_SPAN')
            self.cell = (rows, cols, [])

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[2].append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            rows, cols, parts = self.cell; column = 0
            while (self.row, column) in self.grid:
                column += 1
            if column + cols > 32:
                raise CloudError('PRICING_TABLE_LIMIT')
            value = ' '.join(''.join(parts).split())
            for row in range(self.row, self.row + rows):
                for col in range(column, column + cols):
                    if (row, col) in self.grid:
                        raise CloudError('AMBIGUOUS_PRICING_TABLE')
                    self.grid[row, col] = value
            self.cell = None
        elif tag == 'table' and self.grid is not None:
            height = max((r for r, _ in self.grid), default=-1) + 1
            width = max((c for _, c in self.grid), default=-1) + 1
            self.tables.append([[self.grid.get((r, c), '') for c in range(width)] for r in range(height)])
            self.grid = None


def verify_mapping(html_bytes):
    try:
        parser = Tables(); parser.feed(html_bytes.decode('utf-8')); parser.close()
    except (UnicodeError, ValueError):
        raise CloudError('INVALID_PRICING_HTML') from None
    candidates = [(table, row) for table in parser.tables for row in table
                  if row and row[0] == 'MODEL' and row.count(API_MODEL) == 1]
    if len(candidates) != 1:
        raise CloudError('MODEL_MAPPING_AMBIGUOUS_OR_MISSING')
    table, header = candidates[0]; col = header.index(API_MODEL)
    versions = [row for row in table if row[0] == 'MODEL VERSION']
    bases = [row for row in table if row[0] == 'BASE URL (OpenAI Format)']
    if len(versions) != 1 or versions[0][col] != VERSION:
        raise CloudError('MODEL_ALIAS_NO_LONGER_0813')
    if len(bases) != 1 or bases[0][col] != 'https://' + API_HOST:
        raise CloudError('API_ORIGIN_MAPPING_CHANGED')
    selected = [header, versions[0], bases[0]]
    for label, ceiling in (('1M INPUT TOKENS(CACHE MISS)', INPUT_RATE),
                           ('1M INPUT TOKENS(CACHE HIT)', INPUT_RATE),
                           ('1M OUTPUT TOKENS', OUTPUT_RATE)):
        rows = [row for row in table if label in row]
        if len(rows) != 2 or sorted('PEAK' if 'PEAK' in row else
                                   'OFF-PEAK' if 'OFF-PEAK' in row else '' for row in rows) != ['OFF-PEAK', 'PEAK']:
            raise CloudError('PRICING_SCHEMA_CHANGED')
        for row in rows:
            if not re.fullmatch(r'\$[0-9]+(?:\.[0-9]+)?', row[col]):
                raise CloudError('PRICING_SCHEMA_CHANGED')
            if not Decimal('0') < Decimal(row[col][1:]) <= ceiling:
                raise CloudError('REVIEWED_PRICE_CEILING_EXCEEDED')
        selected.extend(rows)
    return {'kind': 'provider_public_mapping_only', 'source_url': SOURCE_URL,
            'fetched_at': utc_now(), 'source_sha256': hashlib.sha256(html_bytes).hexdigest(),
            'api_model': API_MODEL, 'model_version': VERSION, 'selected_table_rows': selected,
            'input_peak_ceiling_usd_per_million': str(INPUT_RATE),
            'output_peak_ceiling_usd_per_million': str(OUTPUT_RATE),
            'inference_performed': False, 'weight_revision_attested': False}


@contextmanager
def wall_timeout(seconds):
    # Intended cloud runner is POSIX Python on its main thread; fail closed elsewhere.
    if not hasattr(signal, 'setitimer') or threading.current_thread() is not threading.main_thread():
        raise CloudError('WALL_TIMEOUT_UNAVAILABLE')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise CloudError('EXISTING_ALARM_REFUSED')
    def expired(_signum, _frame):
        raise CloudError('WALL_TIMEOUT')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


class FixedHTTPS:
    """TLS verification on; two fixed origins; no redirect/proxy/cookie/retry machinery."""
    def __init__(self, timeout):
        self.timeout = timeout; self.inference_requests = 0; self.metadata_requests = 0

    def _request(self, host, path, method, body=None, key=None):
        headers = {'Accept': 'application/json' if method == 'POST' else 'text/html',
                   'User-Agent': 'MusicJapan-1650B-proof/1', 'Cache-Control': 'no-cache'}
        if key is not None:
            headers.update({'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        connection = http.client.HTTPSConnection(host, 443, timeout=self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                # Never read/log a failure body; redirects are errors, not followed.
                raise CloudError('HTTP_' + str(response.status))
            wire = response.read(MAX_RESPONSE_BYTES + 1)
            if len(wire) > MAX_RESPONSE_BYTES:
                raise CloudError('RESPONSE_SIZE_LIMIT')
            if key and key.encode('ascii') in wire:
                raise CloudError('AUTH_KEY_REFLECTED_BY_PROVIDER')
            return wire
        except CloudError:
            raise
        except Exception:
            raise CloudError('HTTPS_REQUEST_FAILED') from None
        finally:
            connection.close()

    def mapping(self):
        if self.metadata_requests:
            raise CloudError('METADATA_REQUEST_LIMIT')
        self.metadata_requests += 1
        return self._request(DOC_HOST, DOC_PATH, 'GET')

    def complete(self, body, key):
        if self.inference_requests:
            raise CloudError('INFERENCE_REQUEST_LIMIT')
        self.inference_requests += 1
        return self._request(API_HOST, API_PATH, 'POST', body, key)


def prepare(prompt, *, public, allow_charge, max_input_bytes, max_output_tokens,
            timeout_seconds, max_cost_usd):
    if public is not True or allow_charge is not True:
        raise CloudError('PUBLIC_INPUT_AND_CHARGE_OPT_IN_REQUIRED')
    if (type(max_input_bytes) is not int or not 128 <= max_input_bytes <= 16384
        or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 1024
        or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120):
        raise CloudError('EXPLICIT_INPUT_OUTPUT_TIMEOUT_LIMITS_REQUIRED')
    if not isinstance(max_cost_usd, str) or not re.fullmatch(r'(?:0|1)(?:\.[0-9]{1,8})?', max_cost_usd):
        raise CloudError('INVALID_USD_BUDGET')
    try:
        budget = Decimal(max_cost_usd)
    except InvalidOperation:
        raise CloudError('INVALID_USD_BUDGET') from None
    if not Decimal('0') < budget <= Decimal('1'):
        raise CloudError('INVALID_USD_BUDGET')
    if not isinstance(prompt, str) or not prompt.strip() or any(ord(c) < 32 and c not in '\n\r\t' for c in prompt):
        raise CloudError('PUBLIC_TEXT_REQUIRED')
    body = {'model': API_MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': False, 'max_tokens': max_output_tokens, 'thinking': {'type': 'disabled'}}
    input_bytes = len(canonical(body['messages']))
    if input_bytes > max_input_bytes:
        raise CloudError('INPUT_BYTE_LIMIT')
    # Deliberately pessimistic admission estimate, not a tokenizer or a provider billing cap.
    input_estimate = input_bytes + 1024
    estimate = cost(input_estimate, max_output_tokens)
    if estimate > budget:
        raise CloudError('ADMISSION_ESTIMATE_EXCEEDS_BUDGET')
    return body, {'max_input_bytes': max_input_bytes, 'actual_input_bytes': input_bytes,
                  'max_output_tokens': max_output_tokens, 'timeout_seconds': timeout_seconds,
                  'max_inference_requests': 1, 'requested_max_cost_usd': str(budget),
                  'admission_input_token_estimate': input_estimate, 'admission_estimate_usd': str(estimate),
                  'provider_enforced_dollar_cap': False, 'budget_basis': 'UTF8 bytes plus 1024 template tokens; reviewed peak rates'}


def read_key(key_file=None):
    try:
        if key_file:
            with open(key_file, 'rb') as handle:
                raw = handle.read(514)
            key = raw.decode('ascii').strip()
        else:
            key = os.environ.get('DEEPSEEK_API_KEY', '')
    except Exception:
        raise CloudError('AUTH_CREDENTIAL_UNREADABLE') from None
    if not isinstance(key, str) or not 16 <= len(key) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise CloudError('DEEPSEEK_API_KEY_REQUIRED')
    return key


def response_evidence(response, limits):
    if not isinstance(response, dict) or 'error' in response:
        raise CloudError('INVALID_COMPLETION_RESPONSE')
    # Retain evidence even for a truncated/rejected completion, without reporting success.
    evidence = {k: response.get(k) for k in ('id', 'model', 'created', 'usage')}
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return evidence, 'INVALID_CHOICES'
    choice = choices[0]; message = choice.get('message')
    evidence['finish_reason'] = choice.get('finish_reason')
    evidence['content'] = message.get('content') if isinstance(message, dict) else None
    if response.get('model') not in (API_MODEL, VERSION):
        return evidence, 'WRONG_REPORTED_MODEL'
    if not isinstance(response.get('id'), str) or not response['id'].strip():
        return evidence, 'MISSING_RESPONSE_ID'
    if choice.get('finish_reason') != 'stop':
        return evidence, 'NONFINAL_COMPLETION'
    if (not isinstance(message, dict) or message.get('role') != 'assistant' or message.get('tool_calls')
        or not isinstance(message.get('content'), str) or not message['content'].strip()):
        return evidence, 'TEXT_COMPLETION_REQUIRED'
    usage = response.get('usage')
    if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']):
        return evidence, 'INVALID_USAGE'
    if usage['completion_tokens'] > limits['max_output_tokens']:
        return evidence, 'REPORTED_OUTPUT_LIMIT_EXCEEDED'
    upper_estimate = cost(usage['prompt_tokens'], usage['completion_tokens'])
    evidence['reported_usage_cost_upper_estimate_usd'] = str(upper_estimate)
    if upper_estimate > Decimal(limits['requested_max_cost_usd']):
        return evidence, 'REPORTED_USAGE_EXCEEDS_BUDGET'
    return evidence, None


def contains_secret(value, key):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and key in item:
            return True
        if isinstance(item, dict):
            pending.extend(item.keys()); pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def execute_once(body, limits, key, *, transport=None, checkpoint=lambda _receipt: None):
    if key in body['messages'][0]['content']:
        raise CloudError('AUTH_KEY_IN_PUBLIC_INPUT')
    transport = transport or FixedHTTPS(limits['timeout_seconds'])
    wire = canonical(body); started = time.monotonic()
    receipt = {'schema': 'mj-trillion-cloud-receipt-v1', 'status': 'preflight',
               'target_model': MODEL, 'target_revision': REVISION, 'endpoint': 'https://' + API_HOST + API_PATH,
               'started_at': utc_now(), 'request_sha256': hashlib.sha256(wire).hexdigest(),
               'public_request': body, 'limits': limits, 'mapping_evidence': None, 'response_evidence': None,
               'inference_requests_attempted': 0, 'metadata_requests_attempted': 0,
               'inference_verified': False, 'weight_revision_attested': False,
               'client_retry': False, 'client_fallback': False, 'client_cache': False,
               'provider_prompt_cache_possible': True, 'charge_opt_in': True}
    checkpoint(receipt)
    try:
        with wall_timeout(limits['timeout_seconds']):
            receipt['metadata_requests_attempted'] = 1
            receipt['mapping_evidence'] = verify_mapping(transport.mapping())
            receipt['status'] = 'inference_attempted'; receipt['inference_requests_attempted'] = 1
            receipt['inference_started_at'] = utc_now(); checkpoint(receipt)
            inference_start = time.monotonic()
            raw = transport.complete(wire, key)
            receipt['inference_elapsed_seconds'] = round(time.monotonic() - inference_start, 6)
            receipt['response_sha256'] = hashlib.sha256(raw).hexdigest()
            if key.encode('ascii') in raw:
                raise CloudError('AUTH_KEY_REFLECTED_BY_PROVIDER')
            try:
                response = loads(raw.decode('utf-8'))
            except Exception:
                raise CloudError('INVALID_RESPONSE_JSON') from None
            if contains_secret(response, key):
                raise CloudError('AUTH_KEY_REFLECTED_BY_PROVIDER')
            evidence, error = response_evidence(response, limits)
            receipt['response_evidence'] = evidence
            receipt['inference_verified'] = error is None
            receipt['status'] = 'inference_complete' if error is None else 'inference_response_rejected'
            if error:
                receipt['error_code'] = error
    except CloudError as exc:
        receipt.update(status='failed', error_code=str(exc), inference_verified=False)
    except Exception:
        receipt.update(status='failed', error_code='CLOUD_EXECUTION_FAILED', inference_verified=False)
    finally:
        receipt['finished_at'] = utc_now(); receipt['elapsed_seconds'] = round(time.monotonic() - started, 6)
        receipt['charge_may_have_occurred'] = receipt['inference_requests_attempted'] == 1
        checkpoint(receipt)
    return receipt


def plan():
    return {'status': 'plan_only', 'network_requests': 0, 'inference_verified': False,
            'target_model': MODEL, 'target_revision': REVISION, 'api_model': API_MODEL,
            'endpoint': 'https://' + API_HOST + API_PATH, 'mapping_source': SOURCE_URL,
            'mapping_checked': False, 'weight_revision_attested': False,
            'execution_policy': 'free_only', 'paid_inference_enabled': False,
            'execute_error_code': 'PAID_INFERENCE_DISABLED_BY_FREE_ONLY_POLICY',
            'max_inference_requests': 0, 'client_retry': False, 'client_fallback': False, 'client_cache': False,
            'provider_enforced_dollar_cap': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Legacy option; always blocked by the free-only policy')
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--allow-charge', action='store_true')
    for name in ('prompt-file', 'receipt', 'api-key-file', 'max-cost-usd'):
        parser.add_argument('--' + name)
    for name in ('max-input-bytes', 'max-output-tokens', 'timeout-seconds'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(), ensure_ascii=False, indent=2)); return 0
    try:
        # Preserve legacy parsers and isolated contract tests, but the CLI must not
        # read inputs/credentials, create receipts or contact a paid provider.
        raise CloudError('PAID_INFERENCE_DISABLED_BY_FREE_ONLY_POLICY')
        if not args.public or not args.allow_charge or not args.prompt_file or not args.receipt:
            raise CloudError('EXPLICIT_EXECUTION_GUARDS_REQUIRED')
        with open(args.prompt_file, 'rb') as handle:
            prompt = handle.read(16385).decode('utf-8')
        body, limits = prepare(prompt, public=args.public, allow_charge=args.allow_charge,
                               max_input_bytes=args.max_input_bytes, max_output_tokens=args.max_output_tokens,
                               timeout_seconds=args.timeout_seconds, max_cost_usd=args.max_cost_usd)
        key = read_key(args.api_key_file)
        # Exclusive creation prevents accidentally replacing an earlier runtime receipt.
        fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            def checkpoint(receipt):
                handle.seek(0); json.dump(receipt, handle, ensure_ascii=False, indent=2)
                handle.truncate(); handle.flush(); os.fsync(handle.fileno())
            receipt = execute_once(body, limits, key, checkpoint=checkpoint)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt['inference_verified'] else 2
    except CloudError as exc:
        result = {'status': 'blocked', 'error_code': str(exc), 'inference_verified': False}
    except Exception:
        result = {'status': 'blocked', 'error_code': 'LOCAL_INPUT_AUTH_OR_RECEIPT_ERROR', 'inference_verified': False}
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 2


if __name__ == '__main__':
    raise SystemExit(main())
