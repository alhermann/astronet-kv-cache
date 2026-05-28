"""Canonical LongBench F1 scorer.

This is the metric used by the LongBench reference implementation
(Bai et al. 2024, ACL Anthology aclanthology.org/2024.acl-long.172).
It computes token-level F1 between a prediction and one or more gold
answers, with SQuAD-style normalisation:

  1. lowercase
  2. strip articles 'a', 'an', 'the'
  3. strip ASCII + Unicode punctuation
  4. collapse whitespace

The intersection is Counter-based (preserves token multiplicity), not
set-based. For multi-gold-answer datasets the metric is the MAX F1
across all golds.

Replaces the set-intersection F1 in baselines/eval_longbench.py.
"""
import re
import string
from collections import Counter
from typing import List, Sequence, Union


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
# A liberal "punctuation" set that matches the reference LongBench
# normalisation: ASCII punctuation plus a handful of Unicode forms.
_PUNCT = set(string.punctuation) | set(
    ",.?!;:'\"‘’“”、。，．？！"
)


def _normalize(text: str) -> str:
    text = text.lower()
    text = _ARTICLE_RE.sub(" ", text)
    text = "".join(ch if ch not in _PUNCT else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def _f1_single(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 1.0 if not p and not g else 0.0
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: Union[str, Sequence[str]]) -> float:
    """Canonical LongBench F1: max F1 across all gold answers (in %)."""
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    return 100.0 * max(_f1_single(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# Self-test (sanity check that we match the reference behaviour on known
# inputs). Run as `python -m baselines.longbench_canonical_f1`.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # exact match
        ("Paris",                "Paris",        100.0),
        # lowercase, articles, punctuation
        ("the Eiffel Tower.",    "Eiffel Tower", 100.0),
        # partial overlap
        ("Paris, France",        "Paris",         66.67),
        # Counter multiplicity matters
        ("Paris Paris Paris",    "Paris",         50.0),  # P=1/3,R=1,F1=0.5
        # multi-gold MAX semantics
        ("paris",                ["Paris", "Berlin"], 100.0),
        # no overlap
        ("Berlin",               "Paris",          0.0),
    ]
    for pred, gold, expected in cases:
        got = f1_score(pred, gold)
        ok = abs(got - expected) < 0.5
        print(f"  pred={pred!r:30s} gold={gold!r:30s} f1={got:6.2f}  "
              f"expected={expected:6.2f}  {'OK' if ok else 'FAIL'}")
