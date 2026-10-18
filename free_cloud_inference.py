#!/usr/bin/env python3
"""One explicit anonymous request to an existing public model; no paid route.

Default: plan only. Execution requires --execute --public and a new receipt.
No credentials, proxies, redirects, retries, fallback, or response cache are used.
The public operator's own hosting costs and future availability are unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from types import MappingProxyType

MODELS = MappingProxyType({
    "qwen-flash-next": MappingProxyType({
        "host": "apathy-exe-qwen3-8-flash-next.hf.space",
        "source": "Qwen/Qwen3.8-Flash-Next",
        "id": "/root/.cache/huggingface/hub/models--AtomicChat--Qwen3.8-Flash-Next-GGUF/"
              "snapshots/1c1f33abc8e835db3215390e8e14c6cc726f5843/"
              "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/"
              "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf",
    }),
    "deepseek-flash-0731": MappingProxyType({
        "host": "apathy-exe-deepseek-v4-flash-0731.hf.space",
        "source": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "id": "/app/model_unified/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf",
    }),
})
TOTAL_SECONDS = 150
MAX_OUTPUT_TOKENS = 384
MAX_INPUT_BYTES = 8192
MAX_WIRE_BYTES = 3 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
SCHEMA = "mj.free-cloud-inference/1"


class InferenceError(Exception):
    """Only static error codes, never upstream bodies or local paths."""

    def __init__(self, code, evidence=None):
        super().__init__(code)
        self.evidence = {} if evidence is None else evidence


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate_key")
        out[key] = value
    return out


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_input(model, prompt, max_tokens):
    if model not in MODELS:
        raise InferenceError("MODEL_NOT_ALLOWED")
    if (not isinstance(prompt, str) or not prompt.strip() or "\0" in prompt
            or len(prompt.encode("utf-8")) > MAX_INPUT_BYTES):
        raise InferenceError("INPUT_LIMIT_OR_INVALID_TEXT")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise InferenceError("OUTPUT_LIMIT")


def fresh_challenge():
    left, right = 100 + secrets.randbelow(900), 100 + secrets.randbelow(900)
    nonce = secrets.token_hex(8)
    word = secrets.choice(("orbit", "signal", "cobalt", "velvet", "planet", "silver"))
    expected = {"nonce": nonce, "product": left * right, "reverse": word[::-1]}
    prompt = (f'Return ONLY JSON with exactly nonce, product, reverse. Copy nonce "{nonce}" '
              f'exactly. product is the integer {left} multiplied by {right}. '
              f'reverse is "{word}" written backwards. No markdown or explanation.')
    return prompt, expected


def grade(raw, expected):
    try:
        actual = strict_json(raw)
    except (ValueError, TypeError, UnicodeError):
        return {"passed": False, "strict_json": False, "checks": {}}
    shape = isinstance(actual, dict) and set(actual) == set(expected)
    checks = {
        "exact_keys": shape,
        "fresh_nonce": shape and type(actual.get("nonce")) is str and actual["nonce"] == expected["nonce"],
        "arithmetic": shape and type(actual.get("product")) is int and actual["product"] == expected["product"],
        "reverse": shape and type(actual.get("reverse")) is str and actual["reverse"] == expected["reverse"],
    }
    return {"passed": all(checks.values()), "strict_json": True, "checks": checks}


def build_request(model, prompt, max_tokens=MAX_OUTPUT_TOKENS, verify=False):
    validate_input(model, prompt, max_tokens)
    body = {
        "model": MODELS[model]["id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 0.95,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    }
    if model == "qwen-flash-next":
        body["chat_template_kwargs"]["preserve_thinking"] = True
        body["thinking_budget_tokens"] = 64
    if verify:
        body["response_format"] = {"type": "json_object"}
    return body


def validate_catalog(catalog, model):
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise InferenceError("CATALOG_INVALID")
    matches = [entry for entry in catalog["data"]
               if isinstance(entry, dict) and entry.get("id") == MODELS[model]["id"]]
    if len(matches) != 1:
        raise InferenceError("CATALOG_MODEL_MISMATCH")


def parse_sse(lines, expected_model):
    parts = []
    seen_model = finished = done = False
    usage = identifier = None
    received = 0
    for raw in lines:
        if not isinstance(raw, bytes):
            raise InferenceError("SSE_INVALID")
        received += len(raw)
        if received > MAX_WIRE_BYTES or len(raw) > MAX_LINE_BYTES:
            raise InferenceError("SSE_LIMIT")
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeError:
            raise InferenceError("SSE_INVALID") from None
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        try:
            event = strict_json(data)
        except (ValueError, TypeError):
            raise InferenceError("SSE_INVALID") from None
        if not isinstance(event, dict) or event.get("error") is not None:
            raise InferenceError("SSE_UPSTREAM_ERROR")
        if "model" in event:
            if event["model"] != expected_model:
                raise InferenceError("REPORTED_MODEL_MISMATCH")
            seen_model = True
        if "id" in event:
            current = event["id"]
            if not isinstance(current, str) or not 1 <= len(current) <= 256:
                raise InferenceError("SSE_ID_INVALID")
            if identifier is not None and current != identifier:
                raise InferenceError("SSE_ID_CONFLICT")
            identifier = current
        if event.get("usage") is not None:
            candidate = event["usage"]
            keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            if not isinstance(candidate, dict) or any(
                    type(candidate.get(key)) is not int or candidate[key] < 0 for key in keys):
                raise InferenceError("USAGE_INVALID")
            if candidate["prompt_tokens"] + candidate["completion_tokens"] != candidate["total_tokens"]:
                raise InferenceError("USAGE_TOTAL_MISMATCH")
            observed = {key: candidate[key] for key in keys}
            if usage is not None and usage != observed:
                raise InferenceError("USAGE_CONFLICT")
            usage = observed
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise InferenceError("SSE_INVALID")
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict) or type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0:
            raise InferenceError("SSE_INVALID")
        delta = choice.get("delta", {})
        if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("refusal"):
            raise InferenceError("SSE_UNSUPPORTED_OUTPUT")
        content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise InferenceError("SSE_INVALID")
        if finished and (content or reasoning):
            raise InferenceError("SSE_CONTENT_AFTER_FINISH")
        parts.append(content)
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish != "stop":
                raise InferenceError("FINISH_REASON_NOT_STOP", {
                    "raw_answer": "".join(parts), "usage": usage,
                    "completion_id": identifier, "reported_model": expected_model if seen_model else None,
                    "finish_reason": finish if isinstance(finish, str) else None,
                })
            if finished:
                raise InferenceError("SSE_MULTI_TERMINAL")
            finished = True
    answer = "".join(parts)
    if not seen_model or not finished or not done or not identifier or not answer.strip():
        raise InferenceError("SSE_UNFINISHED", {
            "raw_answer": answer, "usage": usage, "completion_id": identifier,
            "reported_model": expected_model if seen_model else None,
            "finish_reason": "stop" if finished else None,
        })
    return {"raw_answer": answer, "usage": usage, "completion_id": identifier,
            "reported_model": expected_model, "finish_reason": "stop"}


class Transport:
    """Fixed direct HTTPS only. The parent process also enforces a wall deadline."""

    def request(self, model, path, body=None, *, timeout):
        if model not in MODELS or path not in ("/v1/models", "/v1/chat/completions"):
            raise InferenceError("TRANSPORT_SCOPE")
        if (path == "/v1/models") != (body is None):
            raise InferenceError("TRANSPORT_SCOPE")
        if not 0 < timeout <= TOTAL_SECONDS:
            raise InferenceError("DEADLINE")
        connection = http.client.HTTPSConnection(MODELS[model]["host"], timeout=min(timeout, 30))
        try:
            headers = {"User-Agent": "MusicJapan-PublicModelResearch/1", "Accept": "application/json" if body is None else "text/event-stream"}
            encoded = None
            if body is not None:
                headers["Content-Type"] = "application/json"
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            connection.request("GET" if body is None else "POST", path, encoded, headers)
            if connection.sock:
                connection.sock.settimeout(timeout)
            response = connection.getresponse()
            if response.status != 200:
                raise InferenceError("UPSTREAM_HTTP_" + str(response.status))
            if body is None:
                raw = response.read(MAX_LINE_BYTES + 1)
                if len(raw) > MAX_LINE_BYTES:
                    raise InferenceError("CATALOG_LIMIT")
                try:
                    return strict_json(raw)
                except (ValueError, TypeError, UnicodeError):
                    raise InferenceError("CATALOG_INVALID") from None
            def lines():
                while True:
                    raw = response.readline(MAX_LINE_BYTES + 1)
                    if not raw:
                        break
                    yield raw
            return parse_sse(lines(), MODELS[model]["id"])
        except InferenceError:
            raise
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            raise InferenceError("CONNECTION_UNAVAILABLE") from None
        finally:
            connection.close()


def token_count_evidence(usage):
    # Provider usage is an observation of its report, not an independent tokenizer measurement.
    return {"completion_token_count_observed": usage is not None,
            "completion_token_count_basis": "provider_usage" if usage is not None else "not_reported"}


def perform(model, prompt, max_tokens, verify, *, transport=None, clock=time.monotonic):
    """Two bounded requests: one catalog GET and at most one inference POST."""
    request = build_request(model, prompt, max_tokens, verify)
    transport = transport or Transport()
    started = clock()
    def remaining():
        value = TOTAL_SECONDS - (clock() - started)
        if value <= 0:
            raise InferenceError("DEADLINE")
        return value
    counts = {"catalog_requests_attempted": 0, "inference_requests_attempted": 0,
              "inference_attempted": False}
    try:
        timeout = min(20, remaining())
        counts["catalog_requests_attempted"] = 1
        catalog = transport.request(model, "/v1/models", timeout=timeout)
        validate_catalog(catalog, model)
        timeout = remaining()
        counts.update(inference_requests_attempted=1, inference_attempted=True)
        result = transport.request(model, "/v1/chat/completions", request, timeout=timeout)
        remaining()  # Expiration wins even if a late response is otherwise correct.
        usage = result.get("usage")
        result.update(token_count_evidence(usage))
        if usage is not None and usage["completion_tokens"] > max_tokens:
            raise InferenceError("REPORTED_OUTPUT_LIMIT_EXCEEDED", dict(result))
        return {**result, **counts}
    except InferenceError as error:
        error.evidence.update(counts)
        error.evidence.update(token_count_evidence(error.evidence.get("usage")))
        raise


def isolated_call(model, prompt, max_tokens, verify, *, deadline):
    """Kill the network child on deadline, including DNS or blocked socket reads.

    The child receives no ambient environment (API keys, proxies, GitHub tokens).
    Only this interpreter and this fixed source file are executable.
    """
    payload = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                          "verify": verify, "public": True}).encode("utf-8")
    process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--_worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env={})
    try:
        budget = deadline - time.monotonic() - 1.0
        if budget <= 0:
            raise InferenceError("DEADLINE")
        try:
            output, _ = process.communicate(payload, timeout=budget)
        except subprocess.TimeoutExpired:
            raise InferenceError("DEADLINE") from None
        if time.monotonic() >= deadline:
            raise InferenceError("DEADLINE")
        if process.returncode != 0 or len(output) > MAX_WIRE_BYTES + 65536:
            raise InferenceError("WORKER_FAILED")
        try:
            value = strict_json(output)
        except (TypeError, ValueError, UnicodeError):
            raise InferenceError("WORKER_INVALID") from None
        if not isinstance(value, dict):
            raise InferenceError("WORKER_INVALID")
        if "error_code" in value:
            code = value["error_code"]
            if not isinstance(code, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{1,80}", code) is None:
                raise InferenceError("WORKER_INVALID")
            evidence = value.get("evidence", {})
            allowed = {"catalog_requests_attempted", "inference_requests_attempted", "inference_attempted",
                       "raw_answer", "usage", "completion_id", "reported_model", "finish_reason",
                       "completion_token_count_observed", "completion_token_count_basis"}
            if not isinstance(evidence, dict) or set(evidence) - allowed:
                raise InferenceError("WORKER_INVALID")
            raise InferenceError(code, evidence)
        return value
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=max(0.05, min(1, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                # Never turn unfinished cleanup or an expired call into success.
                raise InferenceError("WORKER_CLEANUP_TIMEOUT") from None


def _worker():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES * 6 + 4096)
        item = strict_json(raw)
        if not isinstance(item, dict) or set(item) != {"model", "prompt", "max_tokens", "verify", "public"} or item["public"] is not True or type(item["verify"]) is not bool:
            raise InferenceError("WORKER_INPUT_INVALID")
        result = perform(item["model"], item["prompt"], item["max_tokens"], item["verify"])
    except InferenceError as error:
        result = {"error_code": str(error), "evidence": error.evidence}
    except Exception:
        result = {"error_code": "WORKER_FAILED"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


def github_evidence():
    keys = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_OS")
    return {key: os.environ[key][:256] for key in keys if key in os.environ}


def base_receipt(model):
    return {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": model, "expected_reported_model": MODELS[model]["id"],
        "source_model": MODELS[model]["source"], "host": MODELS[model]["host"],
        "github": github_evidence(), "model_identity_basis": "operator_catalog_and_response_report_only",
        "weight_bytes_attested": False, "provider_hosting_cost_known": False,
        "company_credentials_sent": False, "paid_route_available": False,
        "automatic_retry": False, "automatic_fallback": False, "cached_answer_used": False,
        "deadline_seconds": TOTAL_SECONDS, "maximum_output_tokens": MAX_OUTPUT_TOKENS,
        "network_execution_started": False, "inference_attempted": False,
        "catalog_requests_attempted": 0, "inference_requests_attempted": 0,
        "raw_answer": None, "grade": None,
        **token_count_evidence(None),
    }


def read_prompt(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise InferenceError("PROMPT_NOT_REGULAR_FILE")
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise InferenceError("INPUT_LIMIT_OR_INVALID_TEXT")
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        raise InferenceError("PROMPT_NOT_UTF8") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--model", choices=tuple(MODELS), default="deepseek-flash-0731")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--verify", action="store_true", help="Create one fresh synthetic arithmetic/nonce challenge")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({**base_receipt(args.model), "status": "PLAN", "requests": 0,
                          "requires_execution_flags": ["--execute", "--public"],
                          "input_limit_bytes": MAX_INPUT_BYTES}, ensure_ascii=False))
        return 0
    if args.receipt is None:
        print('{"status":"FAILED","error_code":"NEW_RECEIPT_PATH_REQUIRED"}')
        return 2
    try:
        fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        print('{"status":"FAILED","error_code":"RECEIPT_CREATE_FAILED"}')
        return 2
    started = time.monotonic()
    receipt = base_receipt(args.model)
    receipt["status"] = "FAILED"
    try:
        if not args.public:
            raise InferenceError("PUBLIC_INPUT_CONFIRMATION_REQUIRED")
        if not args.verify and args.prompt_file is None:
            raise InferenceError("PUBLIC_INPUT_REQUIRED")
        if args.verify:
            prompt, expected = fresh_challenge()
            receipt["expected"] = expected
        else:
            prompt, expected = read_prompt(args.prompt_file), None
        validate_input(args.model, prompt, args.max_tokens)
        receipt.update(prompt=prompt, prompt_sha256=digest(prompt), requested_output_tokens=args.max_tokens,
                       input_kind="fresh_synthetic_challenge" if args.verify else "owner_declared_public_file")
        # A killed worker may never have reached POST. Unknown is not success.
        receipt.update(network_execution_started=True, inference_attempted=None,
                       catalog_requests_attempted=None, inference_requests_attempted=None)
        result = isolated_call(args.model, prompt, args.max_tokens, args.verify, deadline=started + TOTAL_SECONDS)
        if result.get("reported_model") != MODELS[args.model]["id"] or result.get("finish_reason") != "stop" or not isinstance(result.get("raw_answer"), str) or not result["raw_answer"].strip():
            raise InferenceError("WORKER_RESULT_MISMATCH")
        receipt.update(result)
        receipt["answer_sha256"] = digest(result["raw_answer"])
        receipt["model_catalog_matched"] = True
        receipt["status"] = "GENERATED"
        if args.verify:
            receipt["grade"] = grade(result["raw_answer"], expected)
            receipt["status"] = "GENERATED_AND_GRADED" if receipt["grade"]["passed"] else "GENERATED_GRADE_FAILED"
        if time.monotonic() - started >= TOTAL_SECONDS:
            raise InferenceError("DEADLINE")
    except InferenceError as error:
        receipt.update(error.evidence)
        receipt.update(status="FAILED", error_code=str(error))
    except (OSError, ValueError, TypeError):
        receipt.update(status="FAILED", error_code="LOCAL_OPERATION_FAILED")
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": receipt["status"], "selected_model": args.model,
                      "error_code": receipt.get("error_code"), "elapsed_seconds": receipt["elapsed_seconds"]}))
    return 0 if receipt["status"] in ("GENERATED", "GENERATED_AND_GRADED") else 2


if __name__ == "__main__":
    raise SystemExit(_worker() if sys.argv[1:] == ["--_worker"] else main())
