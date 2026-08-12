"""
Retrieval metrics for the RAG eval harness (keyword-proxy relevance).

We don't have a fully labelled gold set, so each eval case declares the
substring(s) that a *relevant* retrieved chunk should contain. A retrieved doc
is a "hit" if its text contains the expected term(s). From the rank of the
first hit we derive recall@k and MRR.

This is a pragmatic proxy: it is reliable for measuring the *relative* lift
between retrieval configs (dense vs hybrid vs hybrid+rerank) on the same data,
not an absolute quality score. Refine `dataset.json` against your real corpus
to make the numbers more meaningful.

Pure functions, no heavy deps — covered by test_metrics.py (runs in CI).
"""

from __future__ import annotations

from collections.abc import Sequence


def doc_is_hit(doc_text: str, expects: Sequence[str], match_all: bool = False) -> bool:
    """True if `doc_text` contains the expected term(s) (case-insensitive)."""
    text = doc_text.lower()
    terms = [e.lower() for e in expects if e]
    if not terms:
        return False
    if match_all:
        return all(term in text for term in terms)
    return any(term in text for term in terms)


def first_hit_rank(
    doc_texts: Sequence[str],
    expects: Sequence[str],
    match_all: bool = False,
) -> int | None:
    """1-based rank of the first relevant doc, or None if no hit in the list."""
    for i, text in enumerate(doc_texts):
        if doc_is_hit(text, expects, match_all):
            return i + 1
    return None


def recall_at_k(first_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of queries whose first relevant doc appears within the top `k`."""
    if not first_ranks:
        return 0.0
    hits = sum(1 for rank in first_ranks if rank is not None and rank <= k)
    return hits / len(first_ranks)


def mrr(first_ranks: Sequence[int | None]) -> float:
    """Mean Reciprocal Rank — average of 1/rank of the first relevant doc."""
    if not first_ranks:
        return 0.0
    return sum((1.0 / rank) for rank in first_ranks if rank) / len(first_ranks)
