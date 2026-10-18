#!/usr/bin/env python3
"""One-shot real public cloud-model canary, with strict model/JSON/grade evidence.

Uses an existing publicly hosted model, not Music Japan's owned weights.
No credentials or private prompts. No quota circumvention, retries, paid fallback,
scheduler, listening HTTP socket, or Windows/Remote Desktop dependence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

HOST = "apathy-exe-qwen3-8-flash-next.hf.space"
UPSTREAM_ID = (
    "/root/.cache/huggingface/hub/models--AtomicChat--Qwen3.8-Flash-Next-GGUF/"
    "snapshots/1c1f33abc8e835db3215390e8e14c6cc726f5843/"
    "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/"
    "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
)
PUBLIC_MESSAGES = [
    {"role": "system", "content": "Return only a JSON object with exactly product and reverse keys. No markdown."},
    {"role": "user", "content": "Calculate 17 multiplied by 31 and reverse the English word orbit. "
                                "Set product to an integer and reverse to a string."},
]
MAX_RECEIVED = 3 * 1024 * 1024
MAX_LINE = 1024 * 1024


class CanaryError(Exception):
    """Only static failure codes; do not leak upstream response bodies."""


def build_request():
    return {
        "model": UPSTREAM_ID, "messages": PUBLIC_MESSAGES,
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": 384, "temperature": 0.0, "top_p": 0.95,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low", "preserve_thinking": True},
        "thinking_budget_tokens": 64,
        "response_format": {"type": "json_object"},
    }


def validate_catalog(catalog):
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise CanaryError("CATALOG_INVALID")
    hits = [x for x in catalog["data"] if isinstance(x, dict) and x.get("id") == UPSTREAM_ID]
    if len(hits) != 1:
        raise CanaryError("CATALOG_MODEL_MISMATCH")


def parse_sse(chunks):
    seen_model = False
    finished = False
    done = False
    parts = []
    usage = None
    completion_id = None
    bytes_count = 0
    for raw in chunks:
        if not isinstance(raw, bytes):
            raise CanaryError("SSE_INVALID")
        bytes_count += len(raw)
        if bytes_count > MAX_RECEIVED or len(raw) > MAX_LINE:
            raise CanaryError("SSE_LIMIT")
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeError:
            raise CanaryError("SSE_INVALID") from None
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        try:
            x = json.loads(payload)
        except (ValueError, TypeError):
            raise CanaryError("SSE_INVALID") from None
        if not isinstance(x, dict) or x.get("error") is not None:
            raise CanaryError("SSE_ERROR")
        if "model" in x:
            if x["model"] != UPSTREAM_ID:
                raise CanaryError("REPORTED_MODEL_MISMATCH")
            seen_model = True
        identifier = x.get("id")
        if identifier:
            if completion_id is not None and completion_id != identifier:
                raise CanaryError("SSE_ID_CONFLICT")
            completion_id = identifier
        if "usage" in x and x["usage"] is not None:
            candidate = x["usage"]
            if not isinstance(candidate, dict) or any(
                type(candidate.get(k)) is not int or candidate[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise CanaryError("INVALID_USAGE")
            usage = {k: candidate[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        choices = x.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise CanaryError("SSE_INVALID")
        if not choices:
            continue
        c = choices[0]
        if not isinstance(c, dict) or c.get("index", 0) != 0:
            raise CanaryError("SSE_INVALID")
        delta = c.get("delta", {})
        if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("refusal"):
            raise CanaryError("SSE_UNSUPPORTED_OUTPUT")
        text = delta.get("content") or ""
        if not isinstance(text, str):
            raise CanaryError("SSE_INVALID")
        if finished and text:
            raise CanaryError("SSE_CONTENT_AFTER_FINISH")
        parts.append(text)
        reason = c.get("finish_reason")
        if reason is not None:
            if reason != "stop":
                raise CanaryError("FINISH_REASON_NOT_STOP")
            if finished:
                raise CanaryError("SSE_MULTI_TERMINAL")
            finished = True
    if not seen_model or not finished or not done or not "".join(parts).strip():
        raise CanaryError("SSE_UNFINISHED")
    return "".join(parts), usage


def _pairs(values):
    record = {}
    for k, v in values:
        if k in record:
            raise ValueError("duplicate_key")
        record[k] = v
    return record


def grade(raw):
    try:
        answer = json.loads(raw, object_pairs_hook=_pairs,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (TypeError, ValueError):
        return "GENERATED_BUT_INVALID_JSON", False
    if not isinstance(answer, dict) or set(answer) != {"product", "reverse"}:
        return "GENERATED_BUT_FAILED_GRADE", False
    good = type(answer["product"]) is int and answer["product"] == 527 and answer["reverse"] == "tibro"
    return ("GENERATED_AND_GRADED", True) if good else ("GENERATED_BUT_FAILED_GRADE", False)


class Transport:
    """Pinned HTTPS host, no redirects/proxies/credentials and a hard timeout."""
    def request(self, path, body=None, timeout=120):
        if path not in ("/v1/models", "/v1/chat/completions") or (
            (path == "/v1/models") != (body is None)
        ):
            raise CanaryError("TRANSPORT_SCOPE")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 180:
            raise CanaryError("TIMEOUT_INVALID")
        connection = http.client.HTTPSConnection(HOST, timeout=min(timeout, 30))
        deadline = time.monotonic() + timeout
        stopped = threading.Event()

        def stop():
            stopped.set()
            try:
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

        timer = threading.Timer(timeout, stop)
        timer.daemon = True
        timer.start()
        try:
            headers = {"User-Agent": "MusicJapan-PublicSyntheticCanary/1.0",
                       "Accept": "application/json" if body is None else "text/event-stream"}
            encoded = None
            if body is not None:
                encoded = json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf8")
                headers["Content-Type"] = "application/json"
            connection.request("GET" if body is None else "POST", path, body=encoded, headers=headers)
            if connection.sock:
                connection.sock.settimeout(max(0.01, deadline - time.monotonic()))
            result = connection.getresponse()
            if result.status != 200:
                raise CanaryError("UPSTREAM_HTTP_" + str(result.status))
            if body is None:
                data = result.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise CanaryError("CATALOG_LIMIT")
                try:
                    parsed = json.loads(data)
                except ValueError:
                    raise CanaryError("CATALOG_INVALID") from None
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise CanaryError("DEADLINE")
                return parsed
            lines = []
            size = 0
            while True:
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise CanaryError("DEADLINE")
                if connection.sock:
                    connection.sock.settimeout(max(.01, deadline - time.monotonic()))
                line = result.readline(MAX_LINE + 1)
                if not line:
                    break
                size += len(line)
                if size > MAX_RECEIVED or len(line) > MAX_LINE:
                    raise CanaryError("SSE_LIMIT")
                lines.append(line)
                if line.strip() == b"data: [DONE]":
                    break
            if stopped.is_set() or time.monotonic() >= deadline:
                raise CanaryError("DEADLINE")
            return lines
        except CanaryError:
            raise
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            raise CanaryError("DEADLINE" if stopped.is_set() or time.monotonic() >= deadline
                              else "CONNECTION_UNAVAILABLE") from None
        finally:
            timer.cancel()
            connection.close()


def execute(transport=None, *, timeout=120):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 20 <= timeout <= 180:
        raise CanaryError("TIMEOUT_INVALID")
    transport = transport or Transport()
    start = time.monotonic()
    def remaining():
        rest = timeout - (time.monotonic() - start)
        if rest <= 0:
            raise CanaryError("DEADLINE")
        return rest
    catalog = transport.request("/v1/models", timeout=min(20, remaining()))
    validate_catalog(catalog)
    req = build_request()
    chunks = transport.request("/v1/chat/completions", req, timeout=remaining())
    output, usage = parse_sse(chunks)
    if time.monotonic() - start >= timeout:
        raise CanaryError("DEADLINE")
    state, good = grade(output)
    return {
        "schema": "mj.public-cloud-canary/1",
        "status": state, "graded_correct": good,
        "model": UPSTREAM_ID, "source": "unverified_host_reported_model",
        "model_catalog_verified": True, "weights_verified": False,
        "provider_price_usd_verified": False, "company_model_api_credits_used": 0,
        "paid_fallback": False, "automatic_retry": False,
        "answer_sha256": hashlib.sha256(output.encode("utf8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(PUBLIC_MESSAGES, ensure_ascii=False, sort_keys=True).encode("utf8")).hexdigest(),
        "elapsed_seconds": round(time.monotonic()-start, 3),
        "usage": usage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_surface": "github_public_standard_runner" if os.getenv("GITHUB_ACTIONS") else "owner_manual",
        "git_sha": os.environ.get("GITHUB_SHA", "")[:40],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Exactly one public synthetic cloud generation")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--receipt", type=Path, default=Path("cloud-public-canary-receipt.json"))
    args = p.parse_args(argv)
    if not args.live:
        print('{"status":"DRY_RUN","live_model_generation":false}')
        return 0
    try:
        out = execute(timeout=args.timeout)
    except CanaryError as error:
        out = {"schema": "mj.public-cloud-canary/1", "status": "NOT_VERIFIED",
               "error_code": str(error), "automatic_retry": False, "paid_fallback": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                            encoding="utf8")
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0 if out["status"] == "GENERATED_AND_GRADED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
