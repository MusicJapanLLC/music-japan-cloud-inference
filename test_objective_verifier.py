import unittest
from objective_verifier import safe_math_expression,derive_check,verify_answer,apply_verifier
from cloud_infer_frontier import CHALLENGES
class VerifierTests(unittest.TestCase):
    def check(self,text,expected):
        self.assertEqual(derive_check(text)["expected"],expected)
    def test_arithmetic_random_outside_original_benchmark(self):
        self.check("Calculate 314 times 29. Answer only with the integer.","9106")
    def test_existing_multistep_is_corrected_without_gold(self):
        case=next(c for c in CHALLENGES if c["id"]=="math_multistep")
        result=apply_verifier(case["prompt"],"114")
        self.assertEqual(result["final_answer"],"156")
        self.assertEqual(result["source"],"deterministic_tool_correction")
        self.assertEqual(result["verification"]["status"],"REJECTED")
    def test_nested_fraction_and_unary(self):
        self.assertEqual(str(safe_math_expression("-(9+3) / 4")),"-3")
        self.assertEqual(str(safe_math_expression("7 / 2")),"7/2")
    def test_reject_python_injection(self):
        for bad in ('__import__("os").system("calc")','2**1000','5//2','[3][0]','True + 1','1/0'):
            with self.assertRaises((ValueError,SyntaxError,ZeroDivisionError)):
                safe_math_expression(bad)
    def test_linear_equation_exact(self):
        self.check("Solve 7x + 5 = 96. Return x as a number only.","13")
        self.check("Solve 3x - 8 = 13. Number only.","7")
    def test_date_arithmetic(self):
        self.check("Today is Thursday. What day of the week will it be in 10 days? Reply only the day.","Sunday")
        self.check("Today is Monday. What happens in 14 days?","Monday")
    def test_ascii_reverse(self):
        self.check("Reverse the ASCII string abcXYZ exactly, preserving letter case.","ZYXcba")
        self.check("Reverse the ASCII string A7bcZ exactly.","Zcb7A")
    def test_sum_of_squares(self):
        self.check("Sum the squares of integers 1 through 5 inclusive. Return only the number.","55")
        self.check("Sum the squares of integers 3 through 7 inclusive.","135")
    def test_exact_literal(self):
        self.check("Return the exact uppercase word ONLINE. Do not add explanation.","ONLINE")
        self.assertEqual(verify_answer("Return the exact uppercase word ONLINE.","online")["status"],"REJECTED")
    def test_abstain_unseen_general_knowledge(self):
        for question in ("What is the capital of France?","Write a moving poem.","Can you write a secure deployment plan?"):
            self.assertEqual(derive_check(question)["status"],"ABSTAIN")
    def test_protect_unsupported_from_guess(self):
        r=apply_verifier("Write a creative paragraph","some words")
        self.assertEqual(r["source"],"model_unverified")
        self.assertEqual(r["final_answer"],"some words")
    def test_correct_model_kept(self):
        c=next(c for c in CHALLENGES if c["id"]=="math_multistep")
        r=apply_verifier(c["prompt"],"156")
        self.assertEqual(r["source"],"model_with_independent_tool_verification")
    def test_non_numeric_model_rejected(self):
        r=verify_answer("Calculate 5 times 6.","the answer is thirty")
        self.assertEqual(r["status"],"REJECTED")
        self.assertEqual(r["expected"],"30")
    def test_division_requires_nonzero(self):
        self.assertEqual(derive_check("Calculate 4/0.")["status"],"ABSTAIN")
    def test_large_numbers_are_bounded(self):
        self.assertEqual(derive_check("Calculate 99999999999999999 times 2.")["status"],"ABSTAIN")
    def test_grade_does_not_read_embedded_benchmark_answers(self):
        source=__import__("inspect").getsource(__import__("objective_verifier"))
        self.assertNotIn("cloud_infer_frontier",source)
        self.assertNotIn("CHALLENGES",source)
if __name__=="__main__":unittest.main()
