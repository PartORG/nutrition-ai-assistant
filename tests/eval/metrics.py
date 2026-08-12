"""
Retrieval metrics for the RAG eval harness.

Relevance comes from one of two sources, in order of preference:

  relevant_ids  Explicit gold document IDs (a document's `recipe_name`). This is
                a real label: the query is judged against *which* documents came
                back, not against whether their text happens to contain a word.
  expects       Legacy keyword proxy — a doc counts as relevant if its text
                contains the term. Kept so older cases still run, but it is a
                weak signal and structurally favours lexical retrievers like
                BM25, which is exactly the thing being measured.

Metrics: nDCG@k is the primary number because it is rank-sensitive across *all*
relevant documents, not just the first one. recall@k and MRR are retained for
continuity with the previous harness.

Every aggregate reports a bootstrap confidence interval. With a few dozen
queries, a difference of a couple of points is noise, and the old harness had no
way to say so.

Pure functions, no heavy deps — covered by test_metrics.py, which runs in CI.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


def doc_is_hit(doc_text: str, expects: Sequence[str], match_all: bool = False) -> bool:
    """Legacy keyword proxy: True if `doc_text` contains the expected term(s)."""
    text = doc_text.lower()
    terms = [e.lower() for e in expects if e]
    if not terms:
        return False
    if match_all:
        return all(term in text for term in terms)
    return any(term in text for term in terms)


def relevance_by_id(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str],
) -> list[int]:
    """Binary relevance per retrieved doc, matched on document ID.

    Comparison is case- and whitespace-insensitive because recipe names in the
    source CSVs carry inconsistent trailing spaces.
    """
    gold = {r.strip().lower() for r in relevant_ids if r}
    return [1 if (d or "").strip().lower() in gold else 0 for d in retrieved_ids]


def relevance_by_keyword(
    doc_texts: Sequence[str], expects: Sequence[str], match_all: bool = False,
) -> list[int]:
    """Binary relevance per retrieved doc, matched on substring."""
    return [1 if doc_is_hit(t, expects, match_all) else 0 for t in doc_texts]


# ---------------------------------------------------------------------------
# Rank metrics — all take a per-rank relevance vector
# ---------------------------------------------------------------------------


def first_hit_rank(relevance: Sequence[int]) -> int | None:
    """1-based rank of the first relevant doc, or None."""
    for i, rel in enumerate(relevance):
        if rel:
            return i + 1
    return None


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    """Discounted cumulative gain over the top k."""
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(list(relevance)[:k]) if rel
    )


def ndcg_at_k(relevance: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    """nDCG@k — DCG normalised by the best achievable ordering.

    `total_relevant` is how many relevant documents exist in the corpus for this
    query. Without it the ideal ranking is computed from what was retrieved,
    which flatters a retriever that missed relevant docs entirely.
    """
    gains = list(relevance)
    ideal_count = total_relevant if total_relevant is not None else sum(gains)
    if not ideal_count:
        return 0.0
    ideal = dcg_at_k([1] * min(ideal_count, k), k)
    return (dcg_at_k(gains, k) / ideal) if ideal else 0.0


def recall_at_k(relevance: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    """Fraction of the query's relevant documents found in the top k."""
    gains = list(relevance)[:k]
    denom = total_relevant if total_relevant is not None else sum(relevance)
    if not denom:
        return 0.0
    return min(sum(gains) / denom, 1.0)


def reciprocal_rank(relevance: Sequence[int]) -> float:
    """1/rank of the first relevant doc, or 0."""
    rank = first_hit_rank(relevance)
    return (1.0 / rank) if rank else 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-query scores.

    Resamples queries with replacement. With ~40 queries the interval is wide,
    which is the point: it stops a 2-point difference being read as an
    improvement.
    """
    values = list(values)
    if len(values) < 2:
        m = mean(values)
        return (m, m)

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations)
    )
    lo_idx = int((1 - confidence) / 2 * iterations)
    hi_idx = min(int((1 + confidence) / 2 * iterations), iterations - 1)
    return (means[lo_idx], means[hi_idx])


def mrr(first_ranks: Sequence[int | None]) -> float:
    """Mean Reciprocal Rank. Retained for continuity with the old harness."""
    if not first_ranks:
        return 0.0
    return sum((1.0 / rank) for rank in first_ranks if rank) / len(first_ranks)
