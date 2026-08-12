"""
Tests for the retrieval metrics.

These run in CI and need no models, index or data — they are pure arithmetic
over relevance vectors. The point is that a metric bug would silently
misreport whether a retrieval change helped, which is worse than having no
metric at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_ci,
    dcg_at_k,
    doc_is_hit,
    first_hit_rank,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_by_id,
    relevance_by_keyword,
)

# ---------------------------------------------------------------------------
# Relevance construction
# ---------------------------------------------------------------------------

def test_relevance_by_id_marks_gold_documents():
    retrieved = ["Moroccan Chicken Tagine", "Apple Pie", "Nutty Baklava"]
    assert relevance_by_id(retrieved, ["Moroccan Chicken Tagine", "Nutty Baklava"]) == [1, 0, 1]


def test_relevance_by_id_tolerates_case_and_stray_whitespace():
    """Recipe names in the source CSVs carry inconsistent trailing spaces."""
    assert relevance_by_id(["  sabina'S Gnocchi "], ["Sabina's Gnocchi"]) == [1]


def test_relevance_by_id_with_no_gold_is_all_zero():
    assert relevance_by_id(["A", "B"], []) == [0, 0]


def test_relevance_by_keyword_matches_substrings():
    assert relevance_by_keyword(["grilled chicken", "apple pie"], ["chicken"]) == [1, 0]


def test_doc_is_hit_match_all_requires_every_term():
    assert doc_is_hit("chicken and rice", ["chicken", "rice"], match_all=True)
    assert not doc_is_hit("chicken only", ["chicken", "rice"], match_all=True)


# ---------------------------------------------------------------------------
# Rank metrics
# ---------------------------------------------------------------------------

def test_first_hit_rank_is_one_based():
    assert first_hit_rank([0, 0, 1, 0]) == 3
    assert first_hit_rank([1, 0]) == 1
    assert first_hit_rank([0, 0]) is None


def test_reciprocal_rank():
    assert reciprocal_rank([0, 1]) == pytest.approx(0.5)
    assert reciprocal_rank([0, 0]) == 0.0


def test_dcg_discounts_by_position():
    """A hit at rank 1 must be worth more than the same hit at rank 2."""
    assert dcg_at_k([1, 0], 2) > dcg_at_k([0, 1], 2)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 1, 0, 0], 4, total_relevant=2) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_returned():
    assert ndcg_at_k([0, 0, 0], 3, total_relevant=2) == 0.0


def test_ndcg_rewards_ranking_relevant_docs_higher():
    good = ndcg_at_k([1, 0, 0, 0], 4, total_relevant=1)
    bad = ndcg_at_k([0, 0, 0, 1], 4, total_relevant=1)
    assert good > bad


def test_ndcg_penalises_missing_relevant_docs():
    """Without total_relevant the ideal is computed from what came back.

    That flatters a retriever which missed relevant documents entirely — it
    would score 1.0 for finding one of five. Passing total_relevant fixes it.
    """
    found_one_of_five = [1, 0, 0, 0, 0]
    assert ndcg_at_k(found_one_of_five, 5) == pytest.approx(1.0)          # flattering
    assert ndcg_at_k(found_one_of_five, 5, total_relevant=5) < 0.5        # honest


def test_recall_counts_against_the_full_gold_set():
    assert recall_at_k([1, 1, 0], 3, total_relevant=4) == pytest.approx(0.5)


def test_recall_respects_the_cutoff():
    assert recall_at_k([0, 0, 1], 2, total_relevant=1) == 0.0
    assert recall_at_k([0, 0, 1], 3, total_relevant=1) == 1.0


def test_recall_is_capped_at_one():
    """Duplicate gold hits must not push recall above 1.0."""
    assert recall_at_k([1, 1, 1], 3, total_relevant=2) == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_mrr_averages_over_all_queries_including_misses():
    """A query with no hit contributes 0, not nothing — it must not be dropped."""
    assert mrr([1, 2, None]) == pytest.approx((1.0 + 0.5) / 3)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7, 0.5]
    lo, hi = bootstrap_ci(values)
    assert lo <= mean(values) <= hi


def test_bootstrap_ci_is_deterministic():
    """Seeded, so a re-run does not appear to change the result."""
    values = [0.1, 0.9, 0.5, 0.3]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_bootstrap_ci_of_identical_values_is_a_point():
    lo, hi = bootstrap_ci([0.5] * 10)
    assert lo == hi == pytest.approx(0.5)


def test_bootstrap_ci_handles_a_single_query():
    assert bootstrap_ci([0.42]) == (0.42, 0.42)


def test_wider_spread_gives_a_wider_interval():
    tight = bootstrap_ci([0.50, 0.51, 0.49, 0.50, 0.51, 0.49])
    loose = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])
