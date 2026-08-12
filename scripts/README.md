# Operator scripts

Manual scripts for running the system by hand. **These are not tests.**

They previously lived in `test_functionality/` with `test_` prefixes, which made
them look like a test suite — they contained zero assertions and could not fail.
The real tests are in `tests/`, and CI runs those.

Everything here needs a live environment: Ollama running, models pulled, and for
most of them a database and prebuilt vector indexes.

## Index management

Moved to a real command — `nutriai-index`. See the
[Vector index management](../README.md#vector-index-management) section.

```bash
uv run nutriai-index inspect     # what's on disk, and is it current?
uv run nutriai-index rebuild     # full re-embed
```

## Smoke checks

Exercise one component against a live Ollama and print the result for a human to
eyeball. Useful when a model or prompt changes.

| Script | Component |
|---|---|
| `smoke_intent_parser.py` | Intent parsing |
| `smoke_rag_recipe.py` | Recipe RAG retrieval |
| `smoke_rag_medical.py` | Medical RAG constraints |
| `smoke_safety_filter.py` | Safety filter |
| `smoke_search_recipes_tool.py` | Recipe search agent tool |

## Development fixtures

| Script | What it does |
|---|---|
| `add_dummy_user.py` | Creates a test user in the database |
| `chat_repl.py` | Interactive chat with the agent — type `quit` to exit |
