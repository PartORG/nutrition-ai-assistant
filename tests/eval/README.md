# RAG retrieval eval

A tiny harness to **measure** the retrieval upgrades (hybrid search + cross-encoder
reranking + BGE-M3 embeddings) instead of trusting that they help. It reports
`recall@k` and `MRR` for three configs over the **same** index:

| config | what it is |
|---|---|
| `dense-only` | FAISS similarity search (the old behaviour) |
| `hybrid` | dense + BM25, fused with RRF |
| `hybrid+rerank` | hybrid candidates re-scored by the cross-encoder |

Every config returns the same number of docs, so the metrics isolate **ranking
quality** — i.e. whether hybrid/rerank push relevant chunks toward the top.

## Run it

Needs the embedding model + the data files (it builds the FAISS index on first
run). Does **not** need Ollama.

```bash
uv run python tests/eval/run_eval.py

# tune the knobs and re-run to compare:
RAG_FETCH_K=40 uv run python tests/eval/run_eval.py
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 uv run python tests/eval/run_eval.py
```

Example output:

```
=== Recipe RAG  (12 queries) ===
config             recall@5    recall@10       MRR
--------------------------------------------------
dense-only            0.583        0.750     0.471
hybrid                0.667        0.833     0.560
hybrid+rerank         0.833        0.917     0.712
```

(Numbers above are illustrative — run it on your data to get real ones.)

## How relevance is defined

There's no fully labelled gold set, so this uses a **keyword proxy**: each case in
`dataset.json` lists the term(s) a *relevant* retrieved chunk should contain. A
doc is a "hit" if its text contains the term(s); the first hit's rank drives
`recall@k` and `MRR`.

This is reliable for **relative** comparison between configs on the same data,
not an absolute quality score. **Refine `dataset.json` against your real corpus**
(open a few retrieved chunks, confirm the keywords actually mark relevant docs)
to make the numbers trustworthy.

## Files

| file | purpose |
|---|---|
| `dataset.json` | eval cases (`query` → `expects` keywords); edit/extend this |
| `metrics.py` | `recall_at_k`, `mrr`, hit detection — pure functions |
| `run_eval.py` | builds the index once, scores each config, prints the table |
| `test_metrics.py` | CI-safe unit tests for the metric math (`uv run pytest tests/eval`) |
