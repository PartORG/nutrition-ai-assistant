"""CI-safe unit tests for the eval metrics (no models / data needed)."""

from metrics import doc_is_hit, first_hit_rank, mrr, recall_at_k


def test_doc_is_hit_any():
    assert doc_is_hit("Grilled CHICKEN breast with herbs", ["chicken"])
    assert doc_is_hit("tofu stir fry", ["chicken", "tofu"])      # any term
    assert not doc_is_hit("tofu stir fry", ["chicken"])
    assert not doc_is_hit("anything", [])                        # no expects → no hit


def test_doc_is_hit_all():
    assert doc_is_hit("chicken and rice bowl", ["chicken", "rice"], match_all=True)
    assert not doc_is_hit("chicken soup", ["chicken", "rice"], match_all=True)


def test_first_hit_rank():
    docs = ["tofu salad", "beef stew", "chicken pot pie"]
    assert first_hit_rank(docs, ["chicken"]) == 3      # 1-based
    assert first_hit_rank(docs, ["salmon"]) is None
    assert first_hit_rank(["chicken wrap"], ["chicken"]) == 1


def test_recall_at_k():
    ranks = [1, 3, None, 8]
    assert recall_at_k(ranks, 5) == 0.5        # ranks 1,3 within top-5 → 2/4
    assert recall_at_k(ranks, 10) == 0.75      # ranks 1,3,8 → 3/4
    assert recall_at_k([], 5) == 0.0


def test_mrr():
    # (1/1 + 1/2 + 0) / 3 = 0.5
    assert abs(mrr([1, 2, None]) - 0.5) < 1e-9
    assert mrr([]) == 0.0
    assert mrr([1, 1, 1]) == 1.0
