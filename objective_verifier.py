#!/usr/bin/env python3
"""Music Japan independent tool-based verification of narrow reasoning tasks.

The verifier derives a checkable result from the user question itself. It does
NOT read benchmark ground-truth labels and never calls a model or cloud API.
Unsupported questions ABSTAIN, not 'verified' by a model's own confidence.

Supported examples: safe arithmetic, basic linear equations, weekdays,
character reversal, integer square sums, exact requested uppercase strings.
Arithmetic is parsed by a bounded AST with rational arithmetic — no eval.
"""
from __future__ import annotations
import ast
import datetime as dt
from fractions import Fraction
import operator
import re

LIMIT_NODES = 32
LIMIT_DIGITS = 12
LIMIT_VALUE = 10**15
WEEKDAYS = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
BINARY = {ast.Add:operator.add,ast.Sub:operator.sub,
          ast.Mult:operator.mul,ast.Div:operator.truediv}
UNARY = {ast.UAdd:lambda x:x,ast.USub:operator.neg}

def safe_math_expression(expr:str)->Fraction:
    if not isinstance(expr,str) or len(expr)>160:
        raise ValueError("EXPRESSION_OUT_OF_SCOPE")
    s=re.sub(r"\btimes\b", "*", expr, flags=re.I)
    s=re.sub(r"\bdivided by\b", "/", s, flags=re.I)
    if re.search(r"[^0-9\s()+\-*/]",s):
        raise ValueError("UNSAFE_EXPRESSION_SYNTAX")
    if "//" in s or "**" in s or s.count("(")>12:
        raise ValueError("UNSAFE_OPERATOR")
    tree=ast.parse(s,mode="eval")
    if sum(1 for _ in ast.walk(tree))>LIMIT_NODES:
        raise ValueError("EXPRESSION_TOO_DEEP")
    def walk(node,depth=0):
        if depth>12:raise ValueError("TOO_DEEP")
        if isinstance(node,ast.Expression):
            return walk(node.body,depth+1)
        if isinstance(node,ast.Constant) and type(node.value) is int:
            if len(str(abs(node.value)))>LIMIT_DIGITS:
                raise ValueError("NUMBER_TOO_LARGE")
            return Fraction(node.value)
        if isinstance(node,ast.BinOp) and type(node.op) in BINARY:
            lhs,rhs=walk(node.left,depth+1),walk(node.right,depth+1)
            if abs(lhs)>LIMIT_VALUE or abs(rhs)>LIMIT_VALUE:
                raise ValueError("VALUE_TOO_LARGE")
            if isinstance(node.op,ast.Div) and not rhs:
                raise ValueError("DIVISION_BY_ZERO")
            return BINARY[type(node.op)](lhs,rhs)
        if isinstance(node,ast.UnaryOp) and type(node.op) in UNARY:
            return UNARY[type(node.op)](walk(node.operand,depth+1))
        raise ValueError("UNSUPPORTED_AST_NODE")
    answer=walk(tree)
    if abs(answer)>LIMIT_VALUE:
        raise ValueError("VALUE_TOO_LARGE")
    return answer

def format_number(x:Fraction)->str:
    return str(x.numerator) if x.denominator==1 else f"{x.numerator}/{x.denominator}"

# Accepted grammars consume the complete request. Unknown clauses abstain.
_NUMERIC_END = (
    r"(?:[.]?\s*|[.]\s+(?:"
    r"(?:Give|Return|Answer)(?: only)?(?: with)?(?: the| a)? (?:number|integer)(?: only)?"
    r"|Number only"
    r")[.]?)"
)


def _match(pattern: str, question: str):
    return re.fullmatch(pattern, question, flags=re.I)


def _numeric_result(match, value: Fraction, domain: str, proof: str):
    if re.search(r"\binteger\b", match[0], re.I) and value.denominator != 1:
        return None
    return {"status": "CHECKABLE", "domain": domain,
            "expected": format_number(value), "proof": proof}


def derive_check(question: str) -> dict:
    """Derive a reference only when every input instruction is understood.

    This deliberately recognizes a small request language, not arbitrary prose.
    Unsupported modifiers, extra tasks, or unknown output constraints abstain.
    """
    if not isinstance(question, str) or len(question) > 5000:
        return {"status": "ABSTAIN", "reason": "INVALID_QUESTION"}
    s = question.strip()
    m = _match(r"(?:Compute|Calculate)\s+([0-9\s()+*/a-z-]+?)" + _NUMERIC_END, s)
    if m:
        try:
            result = _numeric_result(m, safe_math_expression(m[1].strip()),
                                     "rational_arithmetic", "whitelisted_fraction_ast")
            if result:
                return result
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            pass

    equation_end = r"(?:[.]\s+Return x as a number only[.]?|" + _NUMERIC_END + r")"
    m = _match(r"Solve\s+(-?\d{1,10})x\s*([+-])\s*(\d{1,10})\s*=\s*(-?\d{1,10})" + equation_end, s)
    if m:
        a, sign, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
        if a:
            result = _numeric_result(m, Fraction(c - (b if sign == "+" else -b), a),
                                     "linear_equation", "exact_rational_algebra")
            if result:
                return result

    m = _match(
        r"(?:Today is|Starting from)\s+"
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[.]\s+"
        r"What day of the week will it be in\s+(\d{1,4})\s+days?\?"
        r"(?:\s+Reply (?:only the day|with the English weekday only)[.]?)?", s)
    if m:
        start = next(i for i, v in enumerate(WEEKDAYS) if v.casefold() == m[1].casefold())
        return {"status": "CHECKABLE", "domain": "weekday_modulo",
                "expected": WEEKDAYS[(start + int(m[2])) % 7], "proof": "modulo_7"}

    m = _match(
        r"Reverse the ASCII string\s+([A-Za-z0-9]{1,80})"
        r"(?: exactly(?:, preserving letter case)?)?[.]?"
        r"(?:\s+Only return the reversed string[.]?)?", s)
    if m:
        return {"status": "CHECKABLE", "domain": "ascii_reverse",
                "expected": m[1][::-1], "proof": "reverse_characters"}

    m = _match(r"Sum the squares of integers\s+(-?\d{1,5})\s+through\s+(-?\d{1,5})"
               r"(?: inclusive)?" + _NUMERIC_END, s)
    if m:
        lo, hi = int(m[1]), int(m[2])
        if 0 <= hi - lo <= 1000:
            return {"status": "CHECKABLE", "domain": "sum_of_squares",
                    "expected": str(sum(i * i for i in range(lo, hi + 1))),
                    "proof": "bounded_integer_enumeration"}

    m = _match(
        r"Return the exact (uppercase )?word\s+([A-Za-z]{2,80})[.]?"
        r"(?:\s+Do not add explanation[.]?)?", s)
    if m:
        return {"status": "CHECKABLE", "domain": "exact_literal",
                "expected": m[2].upper() if m[1] else m[2],
                "proof": "literal_from_request"}
    return {"status": "ABSTAIN", "reason": "NO_INDEPENDENT_DETERMINISTIC_CHECK_AVAILABLE"}

def verify_answer(question:str,answer:str)->dict:
    derived=derive_check(question)
    if derived["status"]!="CHECKABLE":
        return {"status":"ABSTAIN","reason":derived["reason"],"independent":False}
    expected=derived["expected"]
    if not isinstance(answer,str) or len(answer)>10000:
        return {"status":"REJECTED","domain":derived["domain"],
                "expected":expected,"independent":True}
    raw=answer.strip()
    if derived["domain"] in ("rational_arithmetic","linear_equation","sum_of_squares"):
        try:
            # Accept fraction, integer or bounded decimal string. No expressions.
            if not re.fullmatch(r"[+-]?\d{1,20}(?:/\d{1,20}|\.\d{1,12})?",raw):
                raise ValueError("NOT_NUMERIC")
            same=Fraction(raw)==Fraction(expected)
        except (ValueError,ZeroDivisionError,OverflowError):
            same=False
    else:
        same=raw==expected
    return {"status":"VERIFIED" if same else "REJECTED",
            "domain":derived["domain"],"expected":expected,
            "method":derived["proof"],"independent":True}

def apply_verifier(question:str,model_answer:str)->dict:
    check=verify_answer(question,model_answer)
    if check["status"]=="REJECTED":
        return {"final_answer":check["expected"],
                "source":"deterministic_tool_correction",
                "model_answer":model_answer,
                "verification":check}
    if check["status"]=="VERIFIED":
        return {"final_answer":model_answer,
                "source":"model_with_independent_tool_verification",
                "verification":check}
    return {"final_answer":model_answer,
            "source":"model_unverified",
            "verification":check}

