"""Offline tests only. No API keys, model credits, external requests or GPU."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import openrouter_free_client as m

def catalog(pricing=None, model=m.ULTRA):
    if pricing is None:
        pricing = {"prompt": "0", "completion": "0", "request": "0", "image": "0"}
    return {"data": [{"id": model, "context_length": 1000000, "pricing": pricing}]}

def answer(model=m.ULTRA):
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": "42"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

class FakeTransport:
    def __init__(self, model=m.ULTRA, pricing=None, result=None, key=None):
        self.model, self.pricing = model, pricing
        self.result = answer(model) if result is None else result
        self.key = {"data": {"is_management_key": False}} if key is None else key
        self.calls = []
    def request(self, path, *, token=None, data=None, timeout=30):
        self.calls.append((path, bool(token), data))
        return {"/models": catalog(self.pricing, self.model), "/key": self.key,
                "/chat/completions": self.result}[path]

class Tests(unittest.TestCase):
    def test_allowlist(self):
        self.assertEqual(len(m.ALLOWLIST), 3)
        for x in m.ALLOWLIST:
            self.assertTrue(x.endswith(":free"))
        for x in ("openrouter/auto", "openai/gpt-4o", "openrouter/free",
                  "nvidia/nemotron-3-ultra-550b-a55b"):
            with self.assertRaises(m.ConnectorError):
                m.build_body(x, "fixture")
    def test_all_pricing_dimensions(self):
        self.assertEqual(m.verified_zero_price(catalog(), m.ULTRA)["verified_price_usd"], 0)
        for x in ("0.001", None, "unknown", "-1", True):
            with self.assertRaises(m.ConnectorError):
                m.verified_zero_price(catalog({"prompt": "0", "completion": "0", "audio": x}), m.ULTRA)
    def test_bad_catalog(self):
        for x in ({}, {"data": []}, {"data": [{"id": m.ULTRA}]},
                  {"data": [{"id": m.ULTRA, "pricing": {"prompt": "0"}}]}):
            with self.assertRaises(m.ConnectorError):
                m.verified_zero_price(x, m.ULTRA)
    def test_body_never_paid(self):
        body = m.build_body(m.ULTRA, "fixture", 30)
        self.assertEqual(body["provider"]["max_price"], {"prompt": 0, "completion": 0})
        self.assertNotIn("fallbacks", body)
        self.assertNotIn("models", body)
    def test_private_rejected_before_network(self):
        fake = FakeTransport()
        with self.assertRaisesRegex(m.ConnectorError, "PUBLIC_INPUT_CONFIRMATION_REQUIRED"):
            m.generate("private", transport=fake)
        self.assertEqual(fake.calls, [])
    def test_inkling_agentic_only(self):
        fake = FakeTransport(model=m.INKLING)
        with self.assertRaisesRegex(m.ConnectorError, "INKLING_REQUIRES_AGENTIC_HARNESS"):
            m.generate("fixture", model=m.INKLING, public=True, transport=fake)
        self.assertEqual(fake.calls, [])
    def test_missing_key_before_completion(self):
        fake = FakeTransport()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaisesRegex(m.ConnectorError, "API_KEY_REQUIRED"):
                m.generate("fixture", public=True, transport=fake,
                    key_file=Path(tempfile.gettempdir()) / "missing-openrouter-key-for-test")
        self.assertEqual([c[0] for c in fake.calls], ["/models"])
    def test_management_key_rejected(self):
        fake = FakeTransport(key={"data": {"is_management_key": True}})
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-" + "a"*36}):
            with self.assertRaisesRegex(m.ConnectorError, "API_KEY_STATUS_REJECTED"):
                m.generate("fixture", public=True, transport=fake)
        self.assertEqual([c[0] for c in fake.calls], ["/models", "/key"])
    def test_success_and_provenance(self):
        fake = FakeTransport()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-" + "a"*36}):
            result = m.generate("fixture", public=True, transport=fake)
        self.assertEqual(result["text"], "42")
        self.assertTrue(result["request_zero_price_ceiling"])
        self.assertFalse(result["self_hosted_weights"])
        self.assertEqual([c[0] for c in fake.calls],
            ["/models", "/key", "/chat/completions"])
        self.assertNotIn("sk-or", str(result))
    def test_mismatch_truncation_tools(self):
        responses = [
            answer("openai/gpt-4o"),
            {**answer(), "choices": [{"message": {"role": "assistant", "content": ""},
                                     "finish_reason": "stop"}]},
            {**answer(), "choices": [{"message": {"role": "assistant", "content": "hi"},
                                     "finish_reason": "length"}]},
            {**answer(), "choices": [{"message": {"role": "assistant", "content": "hi",
                                                 "tool_calls": [{"function": "shell"}]},
                                     "finish_reason": "stop"}]},
        ]
        for x in responses:
            with self.assertRaises(m.ConnectorError):
                m.check_response(x, m.ULTRA)
    def test_redirect_rejected(self):
        with self.assertRaisesRegex(m.ConnectorError, "REDIRECT_BLOCKED"):
            m.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://evil.example")
    def test_prompt_bounds(self):
        for x in ("", " ", "x" * (m.MAX_PROMPT_BYTES+1)):
            with self.assertRaises(m.ConnectorError):
                m.build_body(m.ULTRA, x)
        for n in (0, 2049, True):
            with self.assertRaises(m.ConnectorError):
                m.build_body(m.ULTRA, "hello", n)

if __name__ == "__main__":
    unittest.main()
