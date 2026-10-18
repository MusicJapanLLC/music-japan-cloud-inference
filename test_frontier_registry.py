import json
import tempfile
import unittest
from pathlib import Path
import frontier_registry as cat

class CatalogTests(unittest.TestCase):
    def dummy(self,model="openai/gpt-oss-120b",license_name="apache-2.0"):
        return {
            "id":model,"sha":"a"*40,"gated":False,"private":False,
            "cardData":{"license":license_name},
            "siblings":[
                {"rfilename":"model.safetensors","size":1024,
                 "lfs":{"sha256":"f"*64}},
                {"rfilename":"config.json","size":400},
            ],
        }
    def test_only_official_publishers(self):
        for k,v in cat.MODEL_TARGETS.items():
            self.assertEqual(k.split("/")[0],v["publisher"])
        with self.assertRaises(ValueError):
            cat.validate_registry_url("unknown/unknown-model")
    def test_stable_receipt(self):
        r=cat.receipt({"schema":cat.SCHEMA,"entries":[]})
        self.assertTrue(cat.verify(r))
        r["entries"].append({"model":"fake"})
        self.assertFalse(cat.verify(r))
    def test_metadata_is_not_inference(self):
        r=cat.run(online=False)
        self.assertTrue(cat.verify(r))
        self.assertEqual(r["model_inference_calls"],0)
        self.assertEqual(r["paid_compute_allocations"],0)
        self.assertTrue(all(x["status"]=="NOT_QUERIED" for x in r["entries"]))
    def test_parse_apache_model(self):
        v=cat.summarize("openai/gpt-oss-120b",self.dummy())
        self.assertTrue(v["public_weights_access_claim"])
        self.assertEqual(v["weight_file_count"],1)
        self.assertEqual(v["sha256_weight_file_metadata_count"],1)
        self.assertFalse(v["runtime_tested"])
    def test_custom_license_needs_commercial_review(self):
        d=self.dummy("MiniMaxAI/MiniMax-M3","minimax-community")
        v=cat.summarize("MiniMaxAI/MiniMax-M3",d)
        self.assertTrue(v["license_matches_expected"])
        self.assertTrue(v["commercial_review_required"])
    def test_revoked_license_rejected(self):
        d=self.dummy()
        d["cardData"]["license"]="other-license"
        v=cat.summarize("openai/gpt-oss-120b",d)
        self.assertFalse(v["public_weights_access_claim"])
    def test_private_rejected(self):
        d=self.dummy()
        d["private"]=True
        self.assertFalse(cat.summarize("openai/gpt-oss-120b",d)["public_metadata_accessible"])
    def test_bad_identity_fails(self):
        with self.assertRaises(ValueError):
            cat.summarize("openai/gpt-oss-120b",self.dummy("evil/claim"))
    def test_no_external_api_auth_on_get(self):
        expected=cat.validate_registry_url("openai/gpt-oss-120b")
        class MockHTTP:
            status=200
            def __init__(self): self.sent=False
            def __enter__(self):return self
            def __exit__(self,*x):pass
            def geturl(self):return expected
            def read(self,n):return json.dumps({
                "id":"openai/gpt-oss-120b",
                "sha":"a"*40,
                "siblings":[]
            }).encode()
        def fake_opener(request,timeout):
            self.assertEqual(request.get_method(),"GET")
            self.assertEqual(request.full_url,expected)
            self.assertEqual(timeout,12)
            self.assertNotIn("Authorization",request.headers)
            self.assertNotIn("Cookie",request.headers)
            return MockHTTP()
        self.assertEqual(cat.fetch_public_meta("openai/gpt-oss-120b",fake_opener)["id"],"openai/gpt-oss-120b")
    def test_public_search_with_synthetic_source(self):
        def fake(model):
            license=cat.MODEL_TARGETS[model]["expected_license"]
            return self.dummy(model,license)
        receipt=cat.run(online=True,fetch=fake)
        self.assertTrue(cat.verify(receipt))
        self.assertEqual(len(receipt["entries"]),len(cat.MODEL_TARGETS))
        self.assertTrue(all(x["status"]=="PUBLIC_METADATA_OBSERVED" for x in receipt["entries"]))
        self.assertEqual(receipt["model_inference_calls"],0)
    def test_forged_model_response_is_not_claimed(self):
        def fake(model):
            return self.dummy("evil/wrong-publisher")
        report=cat.run(online=True,fetch=fake)
        self.assertTrue(all(x["status"]=="NOT_VERIFIED" for x in report["entries"]))
    def test_no_bytes_in_registry_receipt(self):
        self.assertFalse(any("token" in k.lower() for k in cat.run().keys()))
    def test_disk_only_not_gpu_allocation(self):
        r=cat.run(online=False)
        self.assertFalse(r["verified_hosted_model_inference"])

if __name__=="__main__":
    unittest.main()
