"""
infrastructure.rag.index_meta - Provenance for a built vector index.

A FAISS index is an opaque pile of floats. Nothing in the files records which
embedding model produced them, so swapping EMBEDDING_MODEL for another model of
the same dimension leaves an index that loads cleanly and returns nonsense —
every query silently retrieves near-random neighbours.

The only guard before this module was a dimension probe (search and catch
AssertionError), which catches 768 -> 1024 but not 1024 -> 1024.

This writes an `index_meta.json` sidecar next to each store recording what built
it, and refuses to load an index that no longer matches the running config.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

META_FILENAME = "index_meta.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexMeta:
    """What produced a vector index."""

    embedding_model: str
    embedding_dim: int
    doc_count: int
    rag_class: str
    built_at: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    # filename -> sha256, so `inspect` can say whether the source data moved on
    source_digests: dict[str, str] | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        embedding_model: str,
        embedding_dim: int,
        doc_count: int,
        rag_class: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        source_files: list[Path] | None = None,
    ) -> IndexMeta:
        return cls(
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            doc_count=doc_count,
            rag_class=rag_class,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_digests=digest_sources(source_files or []),
        )


def meta_file(store_dir: Path) -> Path:
    return Path(store_dir) / META_FILENAME


def digest_sources(paths: list[Path]) -> dict[str, str]:
    """sha256 each source file, keyed by name. Missing files are skipped."""
    digests: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        digests[path.name] = h.hexdigest()
    return digests


def write_meta(store_dir: Path, meta: IndexMeta) -> None:
    target = meta_file(store_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(meta), indent=2) + "\n")
    logger.info(
        "Wrote index provenance: %s (model=%s, dim=%d, docs=%d)",
        target, meta.embedding_model, meta.embedding_dim, meta.doc_count,
    )


def read_meta(store_dir: Path) -> IndexMeta | None:
    """Load provenance, or None if absent/unreadable (e.g. a pre-Phase-2 index)."""
    target = meta_file(store_dir)
    if not target.is_file():
        return None
    try:
        raw = json.loads(target.read_text())
        known = {f for f in IndexMeta.__dataclass_fields__}
        return IndexMeta(**{k: v for k, v in raw.items() if k in known})
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Ignoring unreadable %s: %s", target, e)
        return None


def describe_mismatch(
    existing: IndexMeta,
    *,
    embedding_model: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    source_files: list[Path] | None = None,
) -> str | None:
    """Return a human-readable reason the index is stale, or None if it matches.

    Only compares what actually invalidates an index. doc_count and built_at are
    recorded for diagnostics, not correctness.
    """
    # An index built from a *different set of files* than the config now asks
    # for is stale even when the model and chunking match. This is not
    # hypothetical: the recipe index shipped holding 2,000 cleaned_healthy_meals
    # documents — 39% of it fabricated filler — because the index predated
    # RAG_INCLUDE_SYNTHETIC_MEALS=false and nothing compared the file sets.
    # stale_sources() could not see it: it only checks files present in both.
    if source_files is not None and existing.source_digests is not None:
        want = {p.name for p in source_files}
        had = set(existing.source_digests)
        if extra := sorted(had - want):
            return f"built from {', '.join(extra)}, which the current config excludes"
        if missing := sorted(want - had):
            return f"built without {', '.join(missing)}, which the current config includes"

    if existing.embedding_model != embedding_model:
        return (
            f"built with embedding model '{existing.embedding_model}', "
            f"but the app is configured for '{embedding_model}'"
        )
    if chunk_size is not None and existing.chunk_size not in (None, chunk_size):
        return (
            f"built with chunk_size={existing.chunk_size}, "
            f"but the app is configured for chunk_size={chunk_size}"
        )
    if chunk_overlap is not None and existing.chunk_overlap not in (None, chunk_overlap):
        return (
            f"built with chunk_overlap={existing.chunk_overlap}, "
            f"but the app is configured for chunk_overlap={chunk_overlap}"
        )
    return None


def stale_sources(existing: IndexMeta, source_files: list[Path]) -> list[str]:
    """Names of source files whose contents changed since the index was built.

    Advisory only — changed data means the index is out of date, not invalid,
    so this never blocks a load.
    """
    if not existing.source_digests:
        return []
    current = digest_sources(source_files)
    return sorted(
        name for name, digest in current.items()
        if name in existing.source_digests and existing.source_digests[name] != digest
    )
