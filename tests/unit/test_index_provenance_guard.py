"""
The provenance guard on BaseRAG, exercised without loading any model.

`_verify_index_provenance` is the thing standing between a swapped embedding
model and silently wrong retrieval, so it is tested directly rather than
through initialize() (which would download bge-m3 and read 57 MB of FAISS).
"""

from __future__ import annotations

import logging

import pytest

from domain.exceptions import RAGError
from infrastructure.rag.base_rag import BaseRAG
from infrastructure.rag.index_meta import IndexMeta, write_meta


class _FakeRAG(BaseRAG):
    """Minimal concrete BaseRAG — the abstract ingest hooks are never called here."""

    SYSTEM_PROMPT = "test {input} {context}"

    def _ingest_documents(self):
        return []

    def _ingest_single_file(self, file_path):
        return []


def _rag(tmp_path, *, embedding_model="BAAI/bge-m3", chunk_size=None):
    rag = _FakeRAG(vectorstore_path=str(tmp_path), embedding_model=embedding_model)
    if chunk_size is not None:
        rag.chunk_size = chunk_size
    return rag


def _stamp(tmp_path, **overrides):
    fields = dict(
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        doc_count=100,
        rag_class="_FakeRAG",
    )
    fields.update(overrides)
    write_meta(tmp_path, IndexMeta.build(**fields))


def test_matching_index_loads_without_complaint(tmp_path):
    _stamp(tmp_path)
    _rag(tmp_path)._verify_index_provenance()  # must not raise


def test_swapped_embedding_model_is_refused(tmp_path):
    """The core guard: same dimension, different model, must not load."""
    _stamp(tmp_path, embedding_model="BAAI/bge-m3")

    rag = _rag(tmp_path, embedding_model="intfloat/multilingual-e5-large")
    with pytest.raises(RAGError) as excinfo:
        rag._verify_index_provenance()

    message = str(excinfo.value)
    assert "stale" in message
    assert "nutriai-index rebuild" in message, "the error must say how to fix it"


def test_changed_chunk_size_is_refused(tmp_path):
    _stamp(tmp_path, chunk_size=300, chunk_overlap=50)

    rag = _rag(tmp_path, chunk_size=800)
    with pytest.raises(RAGError, match="stale"):
        rag._verify_index_provenance()


def test_legacy_index_without_meta_is_allowed_but_warns(tmp_path, caplog):
    """Upgrading must not brick an existing deployment — warn, don't raise."""
    with caplog.at_level(logging.WARNING):
        _rag(tmp_path)._verify_index_provenance()

    assert "predates provenance tracking" in caplog.text
    assert "nutriai-index rebuild" in caplog.text


def test_changed_source_data_warns_but_does_not_block(tmp_path, caplog):
    """Stale content is a reason to rebuild, not a reason to refuse to serve."""
    source = tmp_path / "recipes.csv"
    source.write_text("name\nsoup\n")
    _stamp(tmp_path, source_files=[source])
    source.write_text("name\nstew\n")

    rag = _rag(tmp_path)
    rag._source_files = lambda: [source]

    with caplog.at_level(logging.WARNING):
        rag._verify_index_provenance()  # must not raise

    assert "source data changed" in caplog.text
    assert "recipes.csv" in caplog.text
