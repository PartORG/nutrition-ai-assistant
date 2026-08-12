"""
Tests for index provenance.

The defect this guards: a FAISS index records nothing about the model that
built it, so swapping EMBEDDING_MODEL for another model of the *same dimension*
loads cleanly and returns near-random neighbours. The old dimension probe
catches 768 -> 1024 and misses 1024 -> 1024 entirely.
"""

from __future__ import annotations

import json

from infrastructure.rag.index_meta import (
    META_FILENAME,
    IndexMeta,
    describe_mismatch,
    digest_sources,
    read_meta,
    stale_sources,
    write_meta,
)


def _meta(**overrides) -> IndexMeta:
    defaults = dict(
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        doc_count=5090,
        rag_class="RecipeNutritionRAG",
        chunk_size=300,
        chunk_overlap=50,
    )
    defaults.update(overrides)
    return IndexMeta.build(**defaults)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_write_then_read_round_trips(tmp_path):
    write_meta(tmp_path, _meta())
    loaded = read_meta(tmp_path)

    assert loaded is not None
    assert loaded.embedding_model == "BAAI/bge-m3"
    assert loaded.embedding_dim == 1024
    assert loaded.doc_count == 5090
    assert loaded.built_at


def test_missing_meta_reads_as_none(tmp_path):
    """A pre-provenance index must not crash the loader."""
    assert read_meta(tmp_path) is None


def test_corrupt_meta_reads_as_none(tmp_path):
    (tmp_path / META_FILENAME).write_text("{not json")
    assert read_meta(tmp_path) is None


def test_unknown_fields_are_ignored(tmp_path):
    """Forward compatibility: a newer writer's extra keys must not break us."""
    payload = {
        "embedding_model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "doc_count": 1,
        "rag_class": "X",
        "built_at": "2026-08-12T00:00:00+00:00",
        "a_field_from_the_future": True,
    }
    (tmp_path / META_FILENAME).write_text(json.dumps(payload))

    loaded = read_meta(tmp_path)
    assert loaded is not None
    assert loaded.embedding_model == "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# Mismatch detection — the whole point
# ---------------------------------------------------------------------------

def test_matching_config_reports_no_mismatch():
    assert describe_mismatch(
        _meta(), embedding_model="BAAI/bge-m3", chunk_size=300, chunk_overlap=50
    ) is None


def test_same_dimension_model_swap_is_caught():
    """The case the dimension probe cannot see: 1024-d -> different 1024-d."""
    reason = describe_mismatch(
        _meta(embedding_model="BAAI/bge-m3", embedding_dim=1024),
        embedding_model="intfloat/multilingual-e5-large",  # also 1024-d
    )
    assert reason is not None
    assert "bge-m3" in reason
    assert "e5-large" in reason


def test_chunk_size_change_is_caught():
    reason = describe_mismatch(_meta(), embedding_model="BAAI/bge-m3", chunk_size=800)
    assert reason is not None
    assert "300" in reason and "800" in reason


def test_chunk_overlap_change_is_caught():
    reason = describe_mismatch(
        _meta(), embedding_model="BAAI/bge-m3", chunk_overlap=150
    )
    assert reason is not None
    assert "overlap" in reason


def test_chunkless_index_ignores_chunk_params():
    """The recipe corpus is one doc per CSV row and is never split."""
    meta = _meta(chunk_size=None, chunk_overlap=None)
    assert describe_mismatch(
        meta, embedding_model="BAAI/bge-m3", chunk_size=800, chunk_overlap=150
    ) is None


# ---------------------------------------------------------------------------
# Source staleness — advisory, never blocking
# ---------------------------------------------------------------------------

def test_source_digests_detect_edited_data(tmp_path):
    csv = tmp_path / "recipes.csv"
    csv.write_text("name,calories\nsoup,100\n")

    meta = IndexMeta.build(
        embedding_model="m", embedding_dim=8, doc_count=1,
        rag_class="X", source_files=[csv],
    )
    assert stale_sources(meta, [csv]) == []

    csv.write_text("name,calories\nsoup,999\n")
    assert stale_sources(meta, [csv]) == ["recipes.csv"]


def test_stale_sources_empty_when_nothing_recorded():
    assert stale_sources(_meta(source_files=[]), []) == []


def test_digest_skips_missing_files(tmp_path):
    real = tmp_path / "a.csv"
    real.write_text("x")
    digests = digest_sources([real, tmp_path / "gone.csv"])
    assert list(digests) == ["a.csv"]
