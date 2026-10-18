#!/usr/bin/env python3
"""Music Japan public-data-only OpenRouter client. Zero-cost endpoints, no paid fallback."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

API = "https://openrouter.ai/api/v1"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b:free"
INKLING = "thinkingmachines/inkling:free"
LAGUNA = "poolside/laguna-s-2.1:free"
ALLOWLIST = frozenset((ULTRA, INKLING, LAGUNA))
MAX_PROMPT_BYTES = 32768
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_TOKENS = 2048

class ConnectorError(Exception):
    """Only a short error code, never a third-party response body or token."""

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ConnectorError("REDIRECT_BLOCKED")

class Transport:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path, *, token=None, data=None, timeout=30):
        if path not in ("/models", "/key", "/chat/completions"):
            raise ConnectorError("ROUTE_NOT_ALLOWLISTED")
        if ((path == "/models" and (token is not None or data is not None))
            or (path == "/key" and (not token or data is not None))
            or (path == "/chat/completions" and (not token or data is None))):
            raise ConnectorError("INVALID_REQUEST_SHAPE")
        headers = {"Accept": "application/json"}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        encoded = None
        if data is not None:
            try:
                encoded = json.dumps(data, allow_nan=False, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf8")
            except (TypeError, ValueError):
                raise ConnectorError("INVALID_JSON_REQUEST") from None
            headers["Content-Type"] = "application/json"
        req = Request(API + path, data=encoded, headers=headers,
                      method="POST" if encoded is not None else "GET")
        try:
            with self.opener.open(req, timeout=timeout) as res:
                if res.status != 200:
                    raise ConnectorError("UPSTREAM_HTTP_" + str(res.status))
                raw = res.read(MAX_BODY_BYTES + 1)
            if len(raw) > MAX_BODY_BYTES:
                raise ConnectorError("RESPONSE_TOO_LARGE")
            value = json.loads(raw.decode("utf8"))
            if not isinstance(value, dict):
                raise ConnectorError("INVALID_JSON_RESPONSE")
            return value
        except HTTPError as e:
            status = e.code
            e.close()
            raise ConnectorError("UPSTREAM_HTTP_" + str(status)) from None
        except (URLError, TimeoutError, OSError):
            raise ConnectorError("UPSTREAM_CONNECTION_FAILED") from None
        except (ValueError, UnicodeError):
            raise ConnectorError("INVALID_JSON_RESPONSE") from None

def default_key_file():
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) / "MusicJapan/APIForge/secrets/openrouter-free.token" if local
            else Path.home() / ".config/music-japan/openrouter-free.token")

def load_key(key_file=None):
    value = os.environ.get("OPENROUTER_API_KEY", "")
    if not value:
        try:
            value = Path(key_file or default_key_file()).read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            raise ConnectorError("API_KEY_REQUIRED") from None
    if (not value.startswith("sk-or-v1-") or not 25 <= len(value) <= 512
            or any(c.isspace() for c in value)):
        raise ConnectorError("API_KEY_INVALID_FORMAT")
    return value

def verified_zero_price(catalog, model_id):
    if model_id not in ALLOWLIST:
        raise ConnectorError("MODEL_NOT_ALLOWLISTED")
    models = catalog.get("data") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        raise ConnectorError("CATALOG_INVALID")
    matches = [x for x in models if isinstance(x, dict) and x.get("id") == model_id]
    if len(matches) != 1:
        raise ConnectorError("FREE_MODEL_NOT_IN_CATALOG")
    record = matches[0]
    prices = record.get("pricing")
    if not isinstance(prices, dict) or not {"prompt", "completion"}.issubset(prices):
        raise ConnectorError("MODEL_PRICE_NOT_VERIFIABLE")
    for value in prices.values():
        try:
            if isinstance(value, bool) or value is None or Decimal(str(value)) != Decimal(0):
                raise ConnectorError("MODEL_NOT_ZERO_PRICE")
        except (InvalidOperation, ValueError, OverflowError):
            raise ConnectorError("MODEL_PRICE_NOT_VERIFIABLE") from None
    return {"id": model_id, "context_length": record.get("context_length"), "verified_price_usd": 0}

def validate_key_status(info):
    data = info.get("data") if isinstance(info, dict) else None
    if not isinstance(data, dict) or data.get("is_management_key") is True:
        raise ConnectorError("API_KEY_STATUS_REJECTED")
    remaining = data.get("limit_remaining")
    if remaining is not None:
        try:
            if Decimal(str(remaining)) < 0:
                raise ConnectorError("API_KEY_NEGATIVE_BALANCE")
        except (InvalidOperation, ValueError):
            raise ConnectorError("API_KEY_STATUS_REJECTED") from None

def build_body(model_id, prompt, max_tokens=256):
    if model_id not in ALLOWLIST:
        raise ConnectorError("MODEL_NOT_ALLOWLISTED")
    if (not isinstance(prompt, str) or not prompt.strip()
            or len(prompt.encode("utf8")) > MAX_PROMPT_BYTES):
        raise ConnectorError("INVALID_PUBLIC_PROMPT")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise ConnectorError("TOKEN_BUDGET_INVALID")
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "provider": {"max_price": {"prompt": 0, "completion": 0}},
    }

def check_response(response, requested_model):
    if not isinstance(response, dict):
        raise ConnectorError("INVALID_GENERATION_RESPONSE")
    try:
        reported = response["model"]
        choice = response["choices"][0]
        content = choice["message"]["content"]
        reason = choice["finish_reason"]
        usage = response.get("usage")
        if (not isinstance(choice, dict) or not isinstance(content, str) or not content.strip()
                or choice["message"].get("tool_calls")):
            raise ValueError
        if reason != "stop":
            raise ConnectorError("GENERATION_NOT_FINISHED")
        if reported not in (requested_model, requested_model.removesuffix(":free")):
            raise ConnectorError("REPORTED_MODEL_MISMATCH")
    except (KeyError, IndexError, TypeError, ValueError):
        raise ConnectorError("INVALID_GENERATION_RESPONSE") from None
    return {
        "requested_model": requested_model,
        "reported_model": reported,
        "reported_model_exact_match": reported == requested_model,
        "text": content,
        "finish_reason": reason,
        "usage": usage if isinstance(usage, dict) else None,
        "self_hosted_weights": False,
        "zero_price_preflight": True,
        "request_zero_price_ceiling": True,
    }

def generate(prompt, *, model=ULTRA, public=False, agentic_harness=False,
             max_tokens=256, transport=None, key_file=None):
    if public is not True:
        raise ConnectorError("PUBLIC_INPUT_CONFIRMATION_REQUIRED")
    if model == INKLING and agentic_harness is not True:
        raise ConnectorError("INKLING_REQUIRES_AGENTIC_HARNESS")
    body = build_body(model, prompt, max_tokens)
    gateway = transport if transport is not None else Transport()
    verified_zero_price(gateway.request("/models"), model)
    key = load_key(key_file)
    validate_key_status(gateway.request("/key", token=key))
    started = time.monotonic()
    output = gateway.request("/chat/completions", token=key, data=body, timeout=180)
    result = check_response(output, model)
    result.update(
        elapsed_seconds=round(time.monotonic()-started, 3),
        observed_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        prompt_sha256=hashlib.sha256(prompt.encode("utf8")).hexdigest(),
        public_input_confirmed=True)
    return result

def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--models", action="store_true", help="Live free catalog; no key needed")
    action.add_argument("--check-key", action="store_true", help="Verify owner key without inference")
    action.add_argument("--prompt", help="Public synthetic prompt")
    cli.add_argument("--public", action="store_true", help="Confirm no personal/customer data")
    cli.add_argument("--agentic-harness", action="store_true", help="Only inside a real coding/tool agent")
    cli.add_argument("--model", choices=sorted(ALLOWLIST), default=ULTRA)
    cli.add_argument("--max-tokens", type=int, default=256)
    cli.add_argument("--key-file", type=Path)
    cli.add_argument("--json", action="store_true")
    args = cli.parse_args(argv)
    try:
        gateway = Transport()
        if args.models:
            catalog = gateway.request("/models")
            verified = []
            for model_id in sorted(ALLOWLIST):
                try:
                    verified.append(verified_zero_price(catalog, model_id))
                except ConnectorError:
                    continue
            print(json.dumps({"free_models": verified}, ensure_ascii=False))
            return 0
        if args.check_key:
            validate_key_status(gateway.request("/key", token=load_key(args.key_file)))
            print('{"api_key_authenticated":true,"generation_performed":false}')
            return 0
        result = generate(args.prompt, model=args.model, public=args.public,
                          agentic_harness=args.agentic_harness, max_tokens=args.max_tokens,
                          transport=gateway, key_file=args.key_file)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["text"])
        return 0
    except ConnectorError as error:
        print("OPENROUTER_FREE_" + str(error), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
