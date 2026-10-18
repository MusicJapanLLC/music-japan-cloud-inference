"""Offline reasoning ensemble regression; no model inference or external network."""
import unittest
from cloud_probe import signed_record
from cloud_infer_frontier import CHALLENGES
import ensemble_retro as ens

def receipt(profile, replacements=None):
    replacements=replacements or {}
    items=[]
    for item in CHALLENGES:
        answer=replacements.get(item["id"],item["expected"])
        items.append({"id":item["id"],"answer":answer,
                      "status":200,"passed":answer==item["expected"],
                      "seconds":2.0})
    return signed_record({"profile":profile,"results":items,"total":12,
                          "score":sum(x["passed"] for x in items)})

class EnsembleTests(unittest.TestCase):
    def test_majority_is_not_oracle(self):
        records=[
           receipt("baseline-9b",{"math_multistep":"114"}),
           receipt("fable-reconstruction-9b",{"math_multistep":"156"}),
           receipt("openai-20b",{"math_multistep":"114"}),
        ]
        r=ens.analyze(records)
        target=next(x for x in r["cases"] if x["id"]=="math_multistep")
        self.assertEqual(target["voted_answer"],"114")
        self.assertFalse(target["majority_passed"])
        self.assertTrue(target["tool_assisted_passed"])
        self.assertEqual(target["tool_source"],"deterministic_tool_correction")
        self.assertFalse(r["oracle_used_in_selection"])
        self.assertGreater(r["totals"]["majority_plus_tools_correct"],r["totals"]["majority_ensemble_correct"])
    def test_unanimity(self):
        rows=[receipt("baseline-9b"),receipt("fable-reconstruction-9b")]
        d=ens.ensemble_decision(rows,0)
        self.assertEqual(d["status"],"MAJORITY")
        self.assertEqual(d["answer"],"437")
    def test_tie_abstains_for_creative_or_noncheckable(self):
        rows=[
            receipt("baseline-9b",{"japanese":"Ohayo"}),
            receipt("fable-reconstruction-9b",{"japanese":"おはよう"})
        ]
        r=ens.analyze(rows)
        item=next(x for x in r["cases"] if x["id"]=="japanese")
        self.assertEqual(item["decision"],"TIED_ABSTAIN")
        self.assertEqual(item["tool_source"],"model_unverified")
        self.assertFalse(item["tool_assisted_passed"])
    def test_tie_covered_by_mathematical_proof(self):
        rows=[
            receipt("baseline-9b",{"math_product":"wrong"}),
            receipt("fable-reconstruction-9b",{"math_product":"437"})
        ]
        r=ens.analyze(rows)
        item=next(x for x in r["cases"] if x["id"]=="math_product")
        self.assertEqual(item["decision"],"TIED_ABSTAIN")
        self.assertEqual(item["tool_assisted_answer"],"437")
    def test_non_comparable_tasks_refused(self):
        a,b=receipt("baseline-9b"),receipt("fable-reconstruction-9b")
        b["results"][0]["id"]="different"
        with self.assertRaisesRegex(ValueError,"NON_COMPARABLE"):
            ens.analyze([a,b])
    def test_duplicate_profile_refused(self):
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            ens.analyze([receipt("baseline-9b"),receipt("baseline-9b")])
    def test_profile_name_not_invented(self):
        self.assertTrue(ens.SCHEMA.startswith("mj.retro"))
    def test_no_model_calls(self):
        r=ens.analyze([receipt("baseline-9b"),receipt("fable-reconstruction-9b")])
        self.assertEqual(r["new_model_calls"],0)
        self.assertEqual(r["external_model_credits_used"],0)
        self.assertTrue(r["retrospective_only_not_live_service"])

if __name__=="__main__":
    unittest.main()
