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

Relevance is a keyword proxy — refine tests/eval/dataset.json against your corpus.
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
from metrics import first_hit_rank, mrr, recall_at_k  # noqa: E402

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
    ranks = []
    for case in cases:
        docs = retriever.invoke(case["query"])
        texts = [d.page_content for d in docs]
        ranks.append(first_hit_rank(texts, case["expects"], case.get("match_all", False)))
    return ranks


def _print_table(configs, make_retriever, cases):
    cols = "".join(f"{'recall@' + str(k):>12}" for k in K_VALUES)
    header = f"{'config':<16}{cols}{'MRR':>10}"
    print(header)
    print("-" * len(header))
    for name, use_hybrid, reranker in configs:
        ranks = _score(make_retriever(use_hybrid, reranker), cases)
        cells = "".join(f"{recall_at_k(ranks, k):>12.3f}" for k in K_VALUES)
        print(f"{name:<16}{cells}{mrr(ranks):>10.3f}")


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

    print("\nNote: relevance is a keyword proxy (see dataset.json). Compare rows "
          "RELATIVELY — higher recall@5 / MRR for hybrid+rerank means the upgrade helps.")


if __name__ == "__main__":
    main()
