"""
Retrieval eval runner — measures recall@k / MRR for the RAG retrievers across
configs (dense-only vs hybrid vs hybrid+rerank) so you can SEE the lift from the
retrieval upgrades on your own data.

What it does:
  1. Builds the shared embedding model once.
  2. Initializes each RAG (loads the FAISS index, or builds it from data on first
     run). Needs the embedding model + the data files — does NOT need Ollama.
  3. For each config, wraps the SAME index in a retriever and scores it. Every
     config returns the same number of docs (EVAL_RETURN_N), so recall@5 / MRR
     isolate *ranking* quality — i.e. whether hybrid/rerank push relevant chunks
     toward the top.

Run:
    uv run python tests/eval/run_eval.py
    RAG_FETCH_K=40 uv run python tests/eval/run_eval.py     # tune candidate pool
    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 uv run python tests/eval/run_eval.py

Relevance comes from labelled document IDs in tests/eval/dataset.json. nDCG@10
is the headline number and carries a 95% bootstrap CI — with a few dozen queries
a 2-point difference is noise, and the previous harness could not say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ and this directory importable.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_ci,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_by_id,
    relevance_by_keyword,
)

from infrastructure.config import Settings  # noqa: E402
from infrastructure.rag.medical_rag import MedicalRAG  # noqa: E402
from infrastructure.rag.recipe_rag import RecipeNutritionRAG  # noqa: E402
from infrastructure.rag.retrieval import HybridRetriever, Reranker  # noqa: E402
from infrastructure.rag.smart_retriever import SmartRetriever  # noqa: E402

K_VALUES = (5, 10)
EVAL_RETURN_N = max(K_VALUES)  # every config returns this many docs (fair compare)


def _load_cases():
    data = json.loads((Path(__file__).parent / "dataset.json").read_text(encoding="utf-8"))
    return data["cases"]


def _score(retriever, cases):
    """Per-query metric vectors. Kept per-query so a CI can be bootstrapped."""
    per_query: dict[str, list[float]] = {
        "ndcg": [], "rr": [], **{f"recall@{k}": [] for k in K_VALUES},
    }
    for case in cases:
        docs = retriever.invoke(case["query"])

        gold = case.get("relevant_ids")
        if gold:
            ids = [d.metadata.get("recipe_name", "") for d in docs]
            relevance = relevance_by_id(ids, gold)
            total_relevant = len(gold)
        else:
            # Legacy keyword proxy. total_relevant is unknown, so recall is
            # measured against what was retrieved — a weaker, flattering number.
            relevance = relevance_by_keyword(
                [d.page_content for d in docs],
                case.get("expects", []),
                case.get("match_all", False),
            )
            total_relevant = None

        per_query["ndcg"].append(ndcg_at_k(relevance, EVAL_RETURN_N, total_relevant))
        per_query["rr"].append(reciprocal_rank(relevance))
        for k in K_VALUES:
            per_query[f"recall@{k}"].append(recall_at_k(relevance, k, total_relevant))
    return per_query


def _print_table(configs, make_retriever, cases):
    """nDCG with a 95% bootstrap CI, then recall@k and MRR for continuity."""
    ndcg_label = f"nDCG@{EVAL_RETURN_N}"
    cols = "".join(f"{'recall@' + str(k):>11}" for k in K_VALUES)
    header = f"{'config':<16}{ndcg_label:>10}{'95% CI':>18}{cols}{'MRR':>9}"
    print(header)
    print("-" * len(header))

    for name, use_hybrid, reranker in configs:
        scores = _score(make_retriever(use_hybrid, reranker), cases)
        lo, hi = bootstrap_ci(scores["ndcg"])
        cells = "".join(f"{mean(scores[f'recall@{k}']):>11.3f}" for k in K_VALUES)
        print(
            f"{name:<16}{mean(scores['ndcg']):>10.3f}"
            f"{f'[{lo:.3f}, {hi:.3f}]':>18}{cells}{mean(scores['rr']):>9.3f}"
        )

    print(
        f"\n  n={len(cases)} queries. Overlapping CIs mean the difference is not "
        f"distinguishable from noise at this sample size."
    )


def main():
    cfg = Settings.from_env()
    cases = _load_cases()
    recipe_cases = [c for c in cases if c.get("rag", "recipe") == "recipe"]
    medical_cases = [c for c in cases if c.get("rag") == "medical"]

    print(f"Embedding model : {cfg.embedding_model}")
    print(f"Reranker model  : {cfg.rag_reranker_model}")
    print(f"fetch_k         : {cfg.rag_fetch_k}   (return_n={EVAL_RETURN_N})")

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    reranker = Reranker(model_name=cfg.rag_reranker_model, top_n=EVAL_RETURN_N)

    configs = [
        ("dense-only", False, None),
        ("hybrid", True, None),
        ("hybrid+rerank", True, reranker),
    ]

    # ---- Recipe RAG ----
    if recipe_cases:
        rag = RecipeNutritionRAG(
            data_folder=str(cfg.data_dir),
            model_name=cfg.active_llm_model,
            vectorstore_path=str(cfg.recipes_nutrition_vector_path),
            embedding_model=cfg.embedding_model,
            embeddings=embeddings,
            include_synthetic_meals=cfg.rag_include_synthetic_meals,
            llm_provider=cfg.llm_provider,
            ollama_base_url=cfg.ollama_base_url,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
        )
        rag.initialize()
        if rag.vectorstore_recipes is None:
            print("\n[skip] Recipe RAG has no index (no data files found).")
        else:
            print(f"\n=== Recipe RAG  ({len(recipe_cases)} queries) ===")
            _print_table(
                configs,
                lambda uh, rr: SmartRetriever(
                    rag.vectorstore_recipes,
                    k=EVAL_RETURN_N, use_hybrid=uh, reranker=rr, fetch_k=cfg.rag_fetch_k,
                ),
                recipe_cases,
            )

    # ---- Medical RAG ----
    if medical_cases:
        rag = MedicalRAG(
            folder_paths=[str(cfg.pdf_dir)],
            model_name=cfg.active_llm_model,
            vectorstore_path=str(cfg.medical_vectorstore_path),
            embedding_model=cfg.embedding_model,
            embeddings=embeddings,
            llm_provider=cfg.llm_provider,
            ollama_base_url=cfg.ollama_base_url,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
        )
        rag.initialize()
        if rag.vectorstore is None:
            print("\n[skip] Medical RAG has no index (no PDFs found).")
        else:
            print(f"\n=== Medical RAG  ({len(medical_cases)} queries) ===")
            _print_table(
                configs,
                lambda uh, rr: HybridRetriever(
                    rag.vectorstore, final_k=EVAL_RETURN_N, fetch_k=cfg.rag_fetch_k,
                    use_hybrid=uh, reranker=rr,
                ),
                medical_cases,
            )

    print(
        "\nRecipe relevance is by labelled document ID; the medical cases still use "
        "the keyword proxy because those PDFs are private. Read differences between "
        "rows, not absolute values — and only trust a difference whose CIs do not "
        "overlap."
    )


if __name__ == "__main__":
    main()
