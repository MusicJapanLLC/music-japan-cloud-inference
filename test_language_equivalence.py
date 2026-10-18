import unittest
from language_equivalence import normalize,verify_translation,LEXICON

class MultilingualTests(unittest.TestCase):
    def test_both_valid_japanese_morning_registers(self):
        prompt="Translate Good morning into Japanese. Only the translation."
        for phrase in ("おはよう","おはようございます","お早う"):
            self.assertEqual(verify_translation(prompt,phrase)["status"],"VERIFIED")
    def test_other_japanese_greetings(self):
        self.assertEqual(verify_translation("Translate thank you into Japanese.","ありがとうございます")["status"],"VERIFIED")
        self.assertEqual(verify_translation("Translate good evening into Japanese.","こんばんは")["status"],"VERIFIED")
        self.assertEqual(verify_translation("Translate hello into Japanese.","こんにちは")["status"],"VERIFIED")
    def test_korean(self):
        self.assertEqual(verify_translation("Translate thank you into Korean.","감사합니다")["status"],"VERIFIED")
    def test_french(self):
        self.assertEqual(verify_translation("Translate good evening into French.","bonsoir")["status"],"VERIFIED")
    def test_spanish(self):
        self.assertEqual(verify_translation("Translate Good morning into Spanish.","Buenos días")["status"],"VERIFIED")
    def test_german(self):
        self.assertEqual(verify_translation("Translate good morning into German.","Guten Morgen")["status"],"VERIFIED")
    def test_unknown_meaning_never_wrongly_rejected(self):
        self.assertEqual(verify_translation("Translate leverage into French.","levier")["status"],"ABSTAIN")
    def test_unknown_variant_abstains_not_wrong(self):
        self.assertEqual(verify_translation("Translate hello into French.","coucou")["status"],"ABSTAIN")
    def test_not_executing_arbitrary_questions(self):
        self.assertEqual(verify_translation("What is the capital of France?","Paris")["status"],"ABSTAIN")
    def test_unicode_normalization(self):
        self.assertEqual(normalize("  Buenos DÍAS  "),"buenos días")
    def test_trust_boundaries(self):
        self.assertTrue(len(LEXICON)>=12)
        self.assertEqual(verify_translation("a"*5000,"hello")["status"],"ABSTAIN")
        self.assertEqual(verify_translation("Translate hello into French.","a"*1000)["status"],"ABSTAIN")
if __name__=="__main__":
    unittest.main()
