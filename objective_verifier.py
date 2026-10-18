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

def derive_check(question:str)->dict:
    """Return a signed-independent, rules-based reference or explicitly abstain."""
    if not isinstance(question,str) or len(question)>5000:
        return {"status":"ABSTAIN","reason":"INVALID_QUESTION"}
    s=question.strip()
    # General arithmetic instructions with simple operators and nested parentheses.
    m=re.match(r"^(?:Compute|Calculate)\s+(.+?)(?:\.\s|[.]?$)",s,flags=re.I)
    if m:
        try:
            x=safe_math_expression(m.group(1).strip().rstrip("."))
            return {"status":"CHECKABLE","domain":"rational_arithmetic",
                    "expected":format_number(x),"proof":"whitelisted_fraction_ast"}
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):
            pass
    # One-variable integer linear equations. The answer is computed algebraically.
    m=re.search(r"\bSolve\s+(-?\d{1,10})x\s*([+-])\s*(\d{1,10})\s*=\s*(-?\d{1,10})\b",s,re.I)
    if m:
        a,sign,b,c=int(m[1]),m[2],int(m[3]),int(m[4])
        if a:
            value=Fraction(c-(b if sign=="+" else -b),a)
            return {"status":"CHECKABLE","domain":"linear_equation",
                    "expected":format_number(value),"proof":"exact_rational_algebra"}
    # Date arithmetic for weekdays.
    m=re.search(r"\b(?:Today is|Starting from)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b.+?\bin\s+(\d{1,4})\s+days?\b",s,re.I)
    if m:
        start=next(i for i,v in enumerate(WEEKDAYS) if v.casefold()==m[1].casefold())
        return {"status":"CHECKABLE","domain":"weekday_modulo",
                "expected":WEEKDAYS[(start+int(m[2]))%7],"proof":"modulo_7"}
    # ASCII reversal, no Unicode ambiguity or hidden punctuation.
    m=re.search(r"\bReverse the ASCII string\s+([A-Za-z0-9]{1,80})\b",s,re.I)
    if m:
        return {"status":"CHECKABLE","domain":"ascii_reverse",
                "expected":m[1][::-1],"proof":"reverse_characters"}
    # Finite integer series.
    m=re.search(r"\bsum the squares of integers\s+(-?\d{1,5})\s+through\s+(-?\d{1,5})\b",s,re.I)
    if m:
        lo,hi=int(m[1]),int(m[2])
        if 0<=hi-lo<=1000:
            return {"status":"CHECKABLE","domain":"sum_of_squares",
                    "expected":str(sum(i*i for i in range(lo,hi+1))),
                    "proof":"bounded_integer_enumeration"}
    # Exact string constraints can be validated without model judgment.
    m=re.search(r"\b(?:exact\s+uppercase\s+word|exact\s+word)\s+([A-Z]{2,80})\b",s,re.I)
    if m:
        return {"status":"CHECKABLE","domain":"exact_literal",
                "expected":m[1],"proof":"literal_from_request"}
    return {"status":"ABSTAIN","reason":"NO_INDEPENDENT_DETERMINISTIC_CHECK_AVAILABLE"}

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
