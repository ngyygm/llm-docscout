"""Answer scoring for DocScout.

Deterministic, string-based (no model in the loop) so rewards are reproducible
and unit-testable. Normalized exact match with a partial-credit fallback for
short numeric / entity answers. A semantic scorer (e.g. sentence-embedding
similarity) can be dropped in via `score_answer` without changing the reward
engine.
"""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_answer(pred: str, gold: str) -> float:
    """1.0 exact (normalized); 1.0 if pred contains ONLY the gold number (handles
    '90 days' vs gold '90'); 0.5 only when pred is a SUB-phrase of gold (pred token-set
    is a subset of gold's); else 0. Does NOT give partial credit when a wrong answer
    merely mentions the gold token (e.g. 'not 90 but 180' vs gold '90' -> 0.0)."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    # numeric gold: pred is correct iff its ONLY number is the gold number
    gnums = re.findall(r"\d+", g)
    if len(gnums) == 1:
        pnums = re.findall(r"\d+", p)
        if len(pnums) == 1 and pnums[0] == gnums[0]:
            return 1.0  # '90 days' == gold '90'
        return 0.0      # any other number(s) -> wrong
    # non-numeric: partial credit only if pred is a sub-phrase of the gold
    ps, gs = set(p.split()), set(g.split())
    if gs and ps and ps.issubset(gs):
        return 0.5
    return 0.0
