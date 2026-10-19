"""Small audited multilingual equivalence checks (not an LLM translation service).

Provides *independent* reference checks for common low-risk greetings. Never
uses private data, a proprietary translation API, or benchmark answer keys.
Out-of-scope or alternate valid translations ABSTAIN rather than guessing.
"""
from __future__ import annotations
import re
import unicodedata

LEXICON={
    ("good morning","japanese"):{"おはよう","おはようございます","お早う","お早うございます"},
    ("good evening","japanese"):{"こんばんは","今晩は"},
    ("thank you","japanese"):{"ありがとう","ありがとうございます"},
    ("hello","japanese"):{"こんにちは","今日は"},
    ("good night","japanese"):{"おやすみ","おやすみなさい"},
    ("good morning","korean"):{"좋은 아침","좋은 아침이에요","안녕하세요"},
    ("thank you","korean"):{"고마워요","감사합니다","고맙습니다","고마워"},
    ("good night","korean"):{"안녕히 주무세요","잘 자","잘 자요"},
    ("hello","french"):{"bonjour","salut"},
    ("good evening","french"):{"bonsoir"},
    ("thank you","french"):{"merci","merci beaucoup"},
    ("hello","spanish"):{"hola"},
    ("good morning","spanish"):{"buenos días"},
    ("thank you","spanish"):{"gracias","muchas gracias"},
    ("hello","german"):{"hallo","guten tag"},
    ("good morning","german"):{"guten morgen"},
}
QUESTION = re.compile(
    r"^\s*Translate\s+(?P<source>.+?)\s+into\s+(?P<language>Japanese|Korean|French|Spanish|German)\s*[.?!]?"
    r"(?:\s+.*)?$", re.I
)
MAX_TRANSLATION_CHARS=120

def normalize(s:str)->str:
    return unicodedata.normalize("NFKC",s).strip().casefold() if isinstance(s,str) else ""

def verify_translation(question:str, answer:str)->dict:
    if not isinstance(question,str) or not isinstance(answer,str):
        return {"status":"ABSTAIN","domain":"unsupported_input"}
    if len(question)>1000 or len(answer)>MAX_TRANSLATION_CHARS:
        return {"status":"ABSTAIN","domain":"unsupported_length"}
    m=QUESTION.match(question)
    if not m:
        return {"status":"ABSTAIN","domain":"not_curated_translation"}
    phrase=normalize(m.group("source"))
    target=normalize(m.group("language"))
    key=(phrase,target)
    accepted=LEXICON.get(key)
    if not accepted:
        return {"status":"ABSTAIN","domain":"out_of_scope_translation"}
    canon={normalize(value) for value in accepted}
    if normalize(answer) in canon:
        return {"status":"VERIFIED","domain":"curated_multilingual_equivalence",
                "source_phrase":phrase,"language":target,
                "independent":True,"note":"Reference list is narrow; not a general semantic judge"}
    return {"status":"ABSTAIN","domain":"candidate_outside_curated_variants",
            "source_phrase":phrase,"language":target,"independent":False}
