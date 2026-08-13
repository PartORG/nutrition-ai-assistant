"""
BM25Retriever's default preprocessor is `text.split()`, which keeps case and
leaves punctuation glued to words. A recipe then indexes as "Chicken", "Oil,"
and "chopped." while user queries arrive lowercase and unpunctuated, so sparse
retrieval matched almost nothing — and because RRF weights each input list
equally, that noise displaced good dense hits.
"""

from __future__ import annotations

from infrastructure.rag.retrieval import bm25_tokenize


def test_lowercases():
    assert bm25_tokenize("Moroccan Chicken Tagine") == ["moroccan", "chicken", "tagine"]


def test_strips_punctuation_that_would_otherwise_glue_to_words():
    assert bm25_tokenize("2 tbsp Olive Oil, chopped.") == [
        "2", "tbsp", "olive", "oil", "chopped",
    ]


def test_field_labels_do_not_swallow_their_value():
    """"Recipe: Risotto" must yield "recipe" and "risotto", not "Recipe:"."""
    assert bm25_tokenize("Recipe: Risotto\nCuisine: /Main Dishes/") == [
        "recipe", "risotto", "cuisine", "main", "dishes",
    ]


def test_a_query_term_matches_the_document_form():
    """The whole point: query token == document token for the same word."""
    doc = bm25_tokenize("Recipe: Spinach Bacon Quiche\nIngredients:\nEggs, Bacon.")
    query = bm25_tokenize("quiche with bacon")
    assert {"quiche", "bacon"} <= set(doc)
    assert {"quiche", "bacon"} <= set(query)


def test_digits_survive():
    """Quantities and times carry real signal for ingredient queries."""
    assert "350" in bm25_tokenize("Bake at 350 degrees")


def test_empty_and_punctuation_only_are_safe():
    assert bm25_tokenize("") == []
    assert bm25_tokenize("--- ,. !!") == []
