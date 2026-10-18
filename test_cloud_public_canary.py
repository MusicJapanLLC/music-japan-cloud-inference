"""Offline contract tests, no HTTP, tokens, or model credits."""
import copy
import json
import unittest

import cloud_public_canary as c

def event(model=c.UPSTREAM_ID, delta=None, finish=None, usage=None):
    value={"id":"chatcmpl-fixture","model":model,
           "choices":[{"index":0,"delta":delta or {},
                       "finish_reason":finish}]}
    if usage is not None: value["usage"]=usage
    return ("data: "+json.dumps(value)+"\n").encode()

def stream(text='{"product":527,"reverse":"tibro"}',model=c.UPSTREAM_ID,finish="stop"):
    return [event(model,{"content":text}),event(model,{},finish),b"data: [DONE]\n"]

class Fake:
    def __init__(self, catalog=None, reply=None):
        self.calls=[]
        self.catalog=({"data":[{"id":c.UPSTREAM_ID}]} if catalog is None else catalog)
        self.reply=(stream() if reply is None else reply)
    def request(self,path,body=None,timeout=120):
        self.calls.append((path,body))
        if path=="/v1/models":return self.catalog
        if path=="/v1/chat/completions":return self.reply
        raise AssertionError("unallowlisted")

class Tests(unittest.TestCase):
    def test_live_synthetic_success(self):
        fake=Fake()
        out=c.execute(fake,timeout=100)
        self.assertEqual(out["status"],"GENERATED_AND_GRADED")
        self.assertTrue(out["graded_correct"])
        self.assertEqual(out["model"],c.UPSTREAM_ID)
        self.assertEqual(len(fake.calls),2)
        self.assertEqual([z[0] for z in fake.calls],["/v1/models","/v1/chat/completions"])
        self.assertEqual(fake.calls[1][1]["response_format"],{"type":"json_object"})
        self.assertEqual(fake.calls[1][1]["messages"],c.PUBLIC_MESSAGES)
        self.assertNotIn("authorization",json.dumps(fake.calls).lower())
        self.assertFalse(out["paid_fallback"])
    def test_no_credentials_or_private_input(self):
        req=c.build_request()
        self.assertEqual(set(req["messages"][1]),{"role","content"})
        self.assertEqual(req["model"],c.UPSTREAM_ID)
        self.assertFalse(any("secret" in k or "token" in k for k in req))
    def test_catalog_mismatch_blocks_before_generation(self):
        f=Fake(catalog={"data":[{"id":"pretend-1650b"}]})
        with self.assertRaisesRegex(c.CanaryError,"CATALOG_MODEL_MISMATCH"):
            c.execute(f)
        self.assertEqual(len(f.calls),1)
    def test_errored_provider_is_not_retried(self):
        f=Fake()
        def deny(path,body=None,timeout=120):
            f.calls.append((path,body))
            if body is None:return f.catalog
            raise c.CanaryError("UPSTREAM_HTTP_402")
        f.request=deny
        with self.assertRaisesRegex(c.CanaryError,"UPSTREAM_HTTP_402"):
            c.execute(f)
        self.assertEqual(len(f.calls),2)
    def test_model_mismatch(self):
        f=Fake(reply=stream(model="different-model"))
        with self.assertRaisesRegex(c.CanaryError,"REPORTED_MODEL_MISMATCH"):
            c.execute(f)
        self.assertEqual(len(f.calls),2)
    def test_no_done_no_success(self):
        f=Fake(reply=stream()[:-1])
        with self.assertRaisesRegex(c.CanaryError,"SSE_UNFINISHED"):
            c.execute(f)
    def test_length_no_success(self):
        f=Fake(reply=stream(finish="length"))
        with self.assertRaisesRegex(c.CanaryError,"FINISH_REASON_NOT_STOP"):
            c.execute(f)
    def test_format_invalid_no_success(self):
        f=Fake(reply=stream(text="Answer is 527"))
        out=c.execute(f)
        self.assertEqual(out["status"],"GENERATED_BUT_INVALID_JSON")
        self.assertFalse(out["graded_correct"])
    def test_answer_wrong_no_success(self):
        f=Fake(reply=stream(text='{"product":528,"reverse":"tibro"}'))
        out=c.execute(f)
        self.assertEqual(out["status"],"GENERATED_BUT_FAILED_GRADE")
        self.assertFalse(out["graded_correct"])
    def test_no_duplicate_json_keys(self):
        f=Fake(reply=stream(text='{"product":527,"product":528,"reverse":"tibro"}'))
        out=c.execute(f)
        self.assertEqual(out["status"],"GENERATED_BUT_INVALID_JSON")
    def test_output_does_not_contain_model_text(self):
        f=Fake()
        x=c.execute(f)
        self.assertNotIn("tibro",json.dumps(x))
        self.assertRegex(x["answer_sha256"],"^[a-f0-9]{64}$")
    def test_refuse_tool_calls(self):
        f=Fake(reply=[event(delta={"tool_calls":[{"id":"x"}]}),event(finish="stop"),b"data: [DONE]\n"])
        with self.assertRaisesRegex(c.CanaryError,"SSE_UNSUPPORTED_OUTPUT"):
            c.execute(f)
    def test_bad_json_container(self):
        f=Fake(catalog={"data":"wrong"})
        with self.assertRaisesRegex(c.CanaryError,"CATALOG_INVALID"):
            c.execute(f)
    def test_sse_untrusted_usage(self):
        f=Fake(reply=[event(delta={"content":"{}"},usage={"completion_tokens":-1}),
                      event(finish="stop"),b"data: [DONE]\n"])
        with self.assertRaisesRegex(c.CanaryError,"INVALID_USAGE"):
            c.execute(f)
    def test_provider_free_claim_not_asserted(self):
        r=c.execute(Fake())
        self.assertEqual(r["provider_price_usd_verified"],False)
        self.assertEqual(r["company_model_api_credits_used"],0)
        self.assertFalse(r["weights_verified"])

if __name__=="__main__":
    unittest.main()
