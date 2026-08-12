"""
adapters.cli.index_cli - Manage the FAISS vector indexes.

One entry point for building, extending and inspecting the vector stores.
Replaces three scripts that lived in test_functionality/ under test_* names.

    nutriai-index inspect                     # what's on disk, is it current?
    nutriai-index build                       # build anything missing
    nutriai-index rebuild --target medical    # force a full re-embed
    nutriai-index add --target medical a.pdf  # append without a full rebuild

Run these on the host: docker-compose mounts vector_databases read-only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from infrastructure.config import Settings
from infrastructure.rag.index_meta import read_meta, stale_sources

TARGETS = ("recipes", "medical")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG construction
#
# Deliberately not via ServiceFactory: that also builds the agent LLM, the
# intent parser and the safety filter, none of which an index rebuild needs.
# ---------------------------------------------------------------------------

def _build_rag(target: str, cfg: Settings):
    """Construct one RAG, uninitialized. Imports are local — each pulls in torch."""
    if target == "medical":
        from infrastructure.rag.medical_rag import MedicalRAG

        return MedicalRAG(
            folder_paths=[str(cfg.pdf_dir)],
            vectorstore_path=str(cfg.medical_vectorstore_path),
            model_name=cfg.active_llm_model,
            embedding_model=cfg.embedding_model,
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
            ollama_base_url=cfg.ollama_base_url,
            llm_provider=cfg.llm_provider,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
        )

    from infrastructure.rag.recipe_rag import RecipeNutritionRAG

    return RecipeNutritionRAG(
        data_folder=str(cfg.data_dir),
        vectorstore_path=str(cfg.recipes_nutrition_vector_path),
        model_name=cfg.active_llm_model,
        embedding_model=cfg.embedding_model,
        ollama_base_url=cfg.ollama_base_url,
        llm_provider=cfg.llm_provider,
        openai_api_key=cfg.openai_api_key,
        groq_api_key=cfg.groq_api_key,
    )


def _meta_dirs(cfg: Settings) -> dict[str, Path]:
    return {
        "recipes": Path(cfg.recipes_nutrition_vector_path),
        "medical": Path(cfg.medical_vectorstore_path),
    }


def _resolve_targets(requested: str) -> list[str]:
    return list(TARGETS) if requested == "all" else [requested]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_inspect(args, cfg: Settings) -> int:
    """Report what's on disk. Loads no models — safe and instant."""
    print(f"Configured embedding model: {cfg.embedding_model}")
    print(f"Configured chunking:        size={cfg.rag_chunk_size} overlap={cfg.rag_chunk_overlap}")
    print()

    exit_code = 0
    for name, path in _meta_dirs(cfg).items():
        print(f"── {name}  ({path})")
        if not path.exists():
            print("   NOT BUILT — run `nutriai-index build`\n")
            exit_code = 1
            continue

        meta = read_meta(path)
        if meta is None:
            print("   no index_meta.json — predates provenance tracking.")
            print("   Run `nutriai-index rebuild` to stamp it.\n")
            exit_code = 1
            continue

        print(f"   built      {meta.built_at}")
        print(f"   model      {meta.embedding_model}  (dim {meta.embedding_dim})")
        if meta.chunk_size is not None:
            print(f"   chunking   size={meta.chunk_size} overlap={meta.chunk_overlap}")
        print(f"   documents  {meta.doc_count}")

        problems = []
        if meta.embedding_model != cfg.embedding_model:
            problems.append(
                f"model mismatch — index has '{meta.embedding_model}', "
                f"config wants '{cfg.embedding_model}'"
            )
        if name == "medical" and meta.chunk_size not in (None, cfg.rag_chunk_size):
            problems.append(
                f"chunk_size mismatch — index has {meta.chunk_size}, "
                f"config wants {cfg.rag_chunk_size}"
            )

        rag_sources: list[Path] = []
        if name == "recipes":
            rag_sources = sorted(Path(cfg.data_dir).glob("*.csv"))
        elif Path(cfg.pdf_dir).is_dir():
            rag_sources = sorted(Path(cfg.pdf_dir).glob("*.pdf"))
        changed = stale_sources(meta, rag_sources)

        if problems:
            for p in problems:
                print(f"   STALE      {p}")
            print("   -> the app will refuse to load this. Run `nutriai-index rebuild`.")
            exit_code = 1
        elif changed:
            print(f"   OUTDATED   source data changed: {', '.join(changed)}")
            print("   -> loads fine, but content is behind. Rebuild when convenient.")
        else:
            print("   OK         matches current config")
        print()

    return exit_code


def _run_build(args, cfg: Settings, *, force: bool) -> int:
    for target in _resolve_targets(args.target):
        verb = "Rebuilding" if force else "Building"
        print(f"{verb} {target} index...")
        rag = _build_rag(target, cfg)
        rag.initialize(force_rebuild=force)
        print(f"  {target} done.")
    print("\nVerify with: nutriai-index inspect")
    return 0


def cmd_build(args, cfg: Settings) -> int:
    return _run_build(args, cfg, force=False)


def cmd_rebuild(args, cfg: Settings) -> int:
    return _run_build(args, cfg, force=True)


def cmd_add(args, cfg: Settings) -> int:
    """Append documents to an existing index without a full re-embed."""
    missing = [p for p in args.paths if not Path(p).is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: no such file: {p}", file=sys.stderr)
        return 2

    rag = _build_rag(args.target, cfg)
    rag.initialize()

    total = 0
    for path in args.paths:
        added = rag.add_documents(str(Path(path).resolve()))
        total += added
        status = f"{added} chunks" if added else "already indexed, skipped"
        print(f"  {Path(path).name}: {status}")

    print(f"\nAdded {total} chunks to the {args.target} index.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nutriai-index",
        description="Build, extend and inspect the FAISS vector indexes.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="show library logging")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="show index provenance and whether it is current")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("build", help="build any index that does not exist yet")
    s.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("rebuild", help="force a full re-embed, discarding the existing index")
    s.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    s.set_defaults(func=cmd_rebuild)

    s = sub.add_parser("add", help="append documents to an existing index")
    s.add_argument("--target", choices=TARGETS, required=True)
    s.add_argument("paths", nargs="+", help="files to ingest")
    s.set_defaults(func=cmd_add)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = Settings.from_env()
    try:
        return args.func(args, cfg)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
