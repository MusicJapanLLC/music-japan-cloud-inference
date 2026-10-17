import tempfile
import json
import unittest
from pathlib import Path
from cloud_probe import signed_record
import compare_frontier as m

def synthetic(profile,score,seconds):
    cases=[{"id":f"test{i}","passed":i<score,"seconds":seconds,
            "status":200,"answer":"sample"} for i in range(12)]
    return signed_record({
        "schema":m.SCHEMA,"profile":profile,"model_id":profile,
        "anthropic_original_model":False,"external_model_credits":0,
        "new_spend_usd":0,"cloud_runner":"github_hosted_public_linux_cpu",
        "inference_confirmed":True,"total":12,"score":score,"results":cases,
    })
class CompareTests(unittest.TestCase):
    def test_never_choose_weaker_large_model(self):
        result=m.rank([synthetic("baseline-9b",10,3),synthetic("dense-27b",8,20)])
        self.assertEqual(result["leader"]["profile"],"baseline-9b")
        self.assertEqual(result["recommendation"],"DO_NOT_AUTO_UPGRADE")
    def test_measurably_better_model_only(self):
        result=m.rank([synthetic("baseline-9b",8,2),synthetic("moe-35b",11,10)])
        self.assertEqual(result["leader"]["profile"],"moe-35b")
        self.assertEqual(result["recommendation"],"RESEARCH_UPGRADE_CANDIDATE")
    def test_ties_choose_faster(self):
        result=m.rank([synthetic("baseline-9b",11,5),synthetic("dense-27b",11,7)])
        self.assertEqual(result["leader"]["profile"],"baseline-9b")
        self.assertEqual(result["recommendation"],"DO_NOT_AUTO_UPGRADE")
    def test_tamper_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"evidence.json"
            row=synthetic("baseline-9b",10,1)
            row["score"]=12
            p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):m.load_receipt(p)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            m.rank([synthetic("baseline-9b",7,2),synthetic("baseline-9b",8,3)])
    def test_empty(self):
        self.assertEqual(m.rank([])["status"],"INSUFFICIENT_EVIDENCE")
    def test_incomplete_blocked(self):
        x=synthetic("moe-35b",10,5)
        x["results"].pop()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"m.json";p.write_text(json.dumps(signed_record(x)))
            with self.assertRaises(ValueError):m.load_receipt(p)
if __name__=="__main__":unittest.main()
