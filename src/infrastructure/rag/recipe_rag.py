"""
infrastructure.rag.recipe_rag - Personalized recipe recommendations.

Implements RecipeRAGPort over the recipe corpus. Output shape is bound to
_RecipeListSchema through the provider's structured-output API, so the model
fills a schema rather than being asked in prose to emit particular JSON.

Food composition lives in the nutrition_facts SQL table, not here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from infrastructure.config import DEFAULT_EMBEDDING_MODEL
from infrastructure.rag.base_rag import BaseRAG
from infrastructure.rag.prompts import RECIPE_SYSTEM_PROMPT
from infrastructure.rag.recipe_ingest import (
    get_batch_meta,
    load_healthy_meals_csv,
    load_recipes_csv,
    load_recipes_data_sample_csv,
    save_batch_meta,
)
from infrastructure.rag.smart_retriever import SmartRetriever

logger = logging.getLogger(__name__)


def _first_json_object(text: str) -> str | None:
    """Return the first complete top-level {...} in `text`, or None.

    Replaces `re.search(r'\\{.*\\}', ..., re.DOTALL)`, which took the first '{'
    to the *last* '}' anywhere in the response. Given two JSON objects — a
    chatty model offering alternatives — that spliced them into one malformed
    blob and lost both. This tracks brace depth instead, ignoring braces inside
    strings, and stops at the first balanced object.

    Returns None for truncated output, since an unbalanced object is not
    recoverable.
    """
    if not text:
        return None

    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


class _NutritionSchema(BaseModel):
    """Per-serving nutrition the model must fill in."""

    calories: float | None = Field(default=None, description="kcal per serving")
    protein_g: float | None = Field(default=None, description="grams of protein")
    carbs_g: float | None = Field(default=None, description="grams of carbohydrate")
    fat_g: float | None = Field(default=None, description="grams of fat")
    fiber_g: float | None = Field(default=None, description="grams of fibre")
    sodium_mg: float | None = Field(default=None, description="milligrams of sodium")
    sugar_g: float | None = Field(default=None, description="grams of sugar")
    saturated_fat_g: float | None = Field(default=None, description="grams of saturated fat")


class _RecipeSchema(BaseModel):
    """One recipe. Field types here replace ~40 lines of defensive coercion.

    `ingredients` used to arrive as dicts about as often as strings, and
    `cook_instructions` as a list about as often as a string — both were patched
    up by hand after the fact. Declaring the types makes the provider enforce
    them.
    """

    name: str = Field(description="Recipe name")
    why_recommended: str = Field(
        default="", description="One sentence referencing a user requirement, max 150 chars"
    )
    servings: int = Field(default=1, description="Number of servings")
    prep_time: str = Field(default="", description="With unit, e.g. '15 minutes'")
    cook_time: str = Field(default="", description="With unit, e.g. '20 minutes'")
    ingredients: list[str] = Field(
        default_factory=list,
        description="Flat list of strings, quantity first and metric: '200g chicken breast'",
    )
    cook_instructions: str = Field(
        default="", description="Numbered steps in ONE string separated by newlines, °C"
    )
    nutrition: _NutritionSchema = Field(default_factory=_NutritionSchema)


class _RecipeListSchema(BaseModel):
    """The chain's output shape."""

    recipes: list[_RecipeSchema] = Field(default_factory=list)


class RecipeNutritionRAG(BaseRAG):
    """RAG system for personalized recipe recommendations over the recipe corpus.

    Nutrition facts moved to the nutrition_facts SQL table — see
    infrastructure.persistence.nutrition_facts_repo.
    """

    LARGE_FILE_THRESHOLD_MB = 50

    # Bind the output shape to the provider instead of asking for it in prose.
    OUTPUT_SCHEMA = _RecipeListSchema

    SYSTEM_PROMPT = RECIPE_SYSTEM_PROMPT

    def __init__(
        self,
        data_folder: str,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.3,
        k: int = 5,
        ollama_base_url: str = "http://localhost:11434/",
        llm_provider: str = "ollama",
        openai_api_key: str | None = None,
        groq_api_key: str | None = None,
        use_hybrid: bool = False,
        reranker=None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings=None,
        include_synthetic_meals: bool = False,
    ):
        # Only Ollama uses the native format="json" parameter.
        # Cloud providers (Groq, OpenAI) rely on the prompt for JSON output.
        llm_format = "json" if llm_provider == "ollama" else None

        super().__init__(
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            temperature=temperature,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            llm_format=llm_format,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            use_hybrid=use_hybrid,
            reranker=reranker,
            rerank_top_n=k,
            embeddings=embeddings,
        )
        self.data_folder = Path(data_folder)
        self.k = k
        self.include_synthetic_meals = include_synthetic_meals

        self.vectorstore_recipes: FAISS | None = None
        self.smart_retriever: SmartRetriever | None = None

    # ================================================================
    # RecipeRAGPort implementation
    # ================================================================

    async def async_ask(self, query: str) -> list:
        """Query the RAG and return structured Recipe objects.

        The chain binds _RecipeListSchema through the provider's structured
        output API, so the shape is enforced rather than requested. The
        string-parsing path below is a fallback for providers that ignore the
        binding and return prose anyway.
        """
        if not self.rag_chain:
            logger.error("RAG chain not initialized — call initialize() first")
            return []

        try:
            response = await asyncio.to_thread(self.rag_chain.invoke, {"input": query})
        except Exception:
            logger.exception("Recipe RAG chain failed")
            return []

        answer = response.get("answer")
        if isinstance(answer, _RecipeListSchema):
            return self._schema_to_recipes(answer)
        if isinstance(answer, dict):
            # Some providers return the validated dict rather than the model.
            return self._schema_to_recipes(_RecipeListSchema(**answer))

        logger.warning(
            "Recipe RAG returned %s instead of the bound schema — falling back to "
            "JSON parsing", type(answer).__name__,
        )
        return self._parse_json_to_recipes(answer if isinstance(answer, str) else "")

    @staticmethod
    def _schema_to_recipes(parsed: _RecipeListSchema) -> list:
        """Map the validated schema onto domain Recipe objects.

        No coercion needed: the types were enforced upstream.
        """
        from domain.models import NutritionValues, Recipe

        return [
            Recipe(
                name=r.name.strip(),
                ingredients=list(r.ingredients),
                nutrition=NutritionValues(**r.nutrition.model_dump()),
                why_recommended=r.why_recommended,
                servings=r.servings or 1,
                prep_time=r.prep_time,
                cook_instructions=r.cook_instructions,
            )
            for r in parsed.recipes
        ]

    @staticmethod
    def _parse_json_to_recipes(raw_json: str) -> list:
        """Parse the LLM JSON output into a list of Recipe domain objects."""

        from domain.models import NutritionValues, Recipe

        blob = _first_json_object(raw_json)
        if blob is None:
            logger.warning("Recipe RAG returned no JSON object: %s", raw_json[:200])
            return []

        try:
            data = json.loads(blob)
        except json.JSONDecodeError as e:
            logger.warning("Recipe RAG JSON parse error: %s", e)
            return []

        recipes = []
        for r in data.get("recipes", []):
            nd = r.get("nutrition") or {}
            nutrition = NutritionValues(
                calories=_to_float(nd.get("calories")),
                protein_g=_to_float(nd.get("protein_g")),
                carbs_g=_to_float(nd.get("carbs_g")),
                fat_g=_to_float(nd.get("fat_g")),
                fiber_g=_to_float(nd.get("fiber_g")),
                sodium_mg=_to_float(nd.get("sodium_mg")),
                sugar_g=_to_float(nd.get("sugar_g")),
                saturated_fat_g=_to_float(nd.get("saturated_fat_g")),
            )
            # cook_instructions may be a string or a list
            instructions = r.get("cook_instructions", "")
            if isinstance(instructions, list):
                instructions = "\n".join(str(s) for s in instructions)

            raw_ingredients = r.get("ingredients", [])
            if isinstance(raw_ingredients, list):
                normalized_ingredients = []
                for ing in raw_ingredients:
                    if isinstance(ing, str):
                        normalized_ingredients.append(ing)
                    elif isinstance(ing, dict):
                        # LLM returned {"name": "...", "amount": "..."} instead of a plain string
                        parts = [str(v) for v in ing.values() if v]
                        normalized_ingredients.append(" ".join(parts))
                    else:
                        normalized_ingredients.append(str(ing))
            else:
                normalized_ingredients = []

            recipes.append(Recipe(
                name=r.get("name", "").strip(),
                ingredients=normalized_ingredients,
                nutrition=nutrition,
                why_recommended=r.get("why_recommended", ""),
                servings=int(r.get("servings") or 1),
                prep_time=str(r.get("prep_time") or ""),
                cook_instructions=instructions,
            ))

        logger.info("Recipe RAG parsed %d recipes from JSON", len(recipes))
        for r in recipes:
            n = r.nutrition
            logger.info(
                "  Recipe '%s' — nutrition: calories=%s, protein=%s, carbs=%s, fat=%s",
                r.name, n.calories, n.protein_g, n.carbs_g, n.fat_g,
            )
        return recipes

    # ================================================================
    # BaseRAG implementations
    # ================================================================

    def _ingest_documents(self) -> list[Document]:
        csv_sources = [
            ("cleaned_recipes.csv", load_recipes_csv, "recipes"),
            ("cleaned_recipes_data_sample.csv", load_recipes_data_sample_csv, "recipes"),
        ]
        if self.include_synthetic_meals:
            csv_sources.append(
                ("cleaned_healthy_meals.csv", load_healthy_meals_csv, "recipes")
            )
        else:
            logger.info(
                "Skipping cleaned_healthy_meals.csv — fabricated filler with no "
                "ingredients (set RAG_INCLUDE_SYNTHETIC_MEALS=true to include it)"
            )

        threshold_bytes = self.LARGE_FILE_THRESHOLD_MB * 1024 * 1024
        all_loaded: dict[str, list[Document]] = {}
        large_files_ingested: list[str] = []

        for filename, loader_fn, collection in csv_sources:
            csv_path = self.data_folder / filename
            if not csv_path.exists():
                logger.warning("CSV not found, skipping: %s", csv_path)
                all_loaded[filename] = []
                continue

            file_size = csv_path.stat().st_size
            if file_size > threshold_bytes:
                logger.info(
                    "Large file detected: %s (%.1f MB) — using batched ingestion",
                    filename, file_size / (1024 * 1024),
                )
                self._ingest_large_csv(str(csv_path), loader_fn, collection)
                large_files_ingested.append(filename)
                all_loaded[filename] = []
            else:
                all_loaded[filename] = loader_fn(str(csv_path))

        recipes1 = all_loaded.get("cleaned_recipes.csv", [])
        recipes2 = all_loaded.get("cleaned_recipes_data_sample.csv", [])
        meals = all_loaded.get("cleaned_healthy_meals.csv", [])

        all_docs = recipes1 + recipes2 + meals
        for doc in all_docs:
            doc.metadata["_collection"] = "recipes"

        logger.info("Loaded %d recipe/meal documents", len(all_docs))
        return all_docs

    def _ingest_single_file(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            logger.warning("File does not exist: %s", file_path)
            return []

        loader_map = {
            "cleaned_recipes.csv": load_recipes_csv,
            "cleaned_recipes_data_sample.csv": load_recipes_data_sample_csv,
            "cleaned_healthy_meals.csv": load_healthy_meals_csv,
        }
        loader_fn = loader_map.get(path.name)
        if loader_fn is None:
            logger.warning("No loader for file: %s", path.name)
            return []
        return loader_fn(file_path)

    # ================================================================
    # Dual vectorstore overrides
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        return (self.vectorstore_path / "recipes_and_meals_db").exists()

    def _index_dimension(self) -> int | None:
        """Both stores share an embedding model, so the recipes store speaks for both."""
        if self.vectorstore_recipes is None:
            return None
        return int(self.vectorstore_recipes.index.d)

    def _source_files(self) -> list[Path]:
        """The CSVs this index is actually built from, for staleness reporting."""
        folder = Path(self.data_folder)
        if not folder.is_dir():
            return []
        names = ["cleaned_recipes.csv", "cleaned_recipes_data_sample.csv"]
        if self.include_synthetic_meals:
            names.append("cleaned_healthy_meals.csv")
        return [folder / n for n in names if (folder / n).is_file()]

    def _embeddings_match_index(self) -> bool:
        """Check BOTH stores — the base hook only looks at self.vectorstore."""
        store = self.vectorstore_recipes
        return store is None or self._index_dim_ok(store)

    def _prepare_rebuild(self) -> None:
        """Wipe both stores + resumable batch-meta so a model switch re-embeds cleanly.

        Without this, switching EMBEDDING_MODEL would leave stale-dimension indices
        (and the batch-ingestion checkpoint would skip re-embedding the rows).
        """
        import shutil

        target = self.vectorstore_path / "recipes_and_meals_db"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        for meta in self.vectorstore_path.glob("*_batch_meta.json"):
            meta.unlink(missing_ok=True)
        self.vectorstore_recipes = None
        logger.info("Cleared recipe/nutrition indices for a clean rebuild")

    def _build_vectorstore(self, documents: list[Document]) -> None:
        logger.info("Building FAISS index (recipes=%d)", len(documents))

        self.vectorstore_recipes = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore_recipes.save_local(str(self.vectorstore_path / "recipes_and_meals_db"))
        logger.info("Vectorstores saved to %s", self.vectorstore_path)

    def _load_vectorstore(self) -> None:
        self.vectorstore_recipes = FAISS.load_local(
            folder_path=str(self.vectorstore_path / "recipes_and_meals_db"),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vectorstore loaded — Recipes: %d", self.vectorstore_recipes.index.ntotal)

    def _setup_retriever(self) -> None:
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            k=self.k,
            use_hybrid=self.use_hybrid,
            reranker=self.reranker,
        )
        self.retriever = self.smart_retriever
        # Pay the BM25 tokenisation cost here, during startup, rather than
        # inside whichever request happens to arrive first after a restart.
        self.smart_retriever.warm_up()
        logger.info(
            "SmartRetriever created (k=%d, hybrid=%s, rerank=%s)",
            self.k, self.use_hybrid, self.reranker is not None,
        )

    # ================================================================
    # Batched CSV ingestion
    # ================================================================

    def _ingest_large_csv(
        self,
        csv_path: str,
        loader_fn,
        collection: str,
        chunksize: int = 5000,
    ) -> None:
        path = Path(csv_path)
        if not path.exists():
            logger.error("CSV file not found: %s", csv_path)
            return

        if self.embeddings is None:
            raise RuntimeError("Call initialize() before _ingest_large_csv()")

        if collection != "recipes":
            raise ValueError(f"Unknown collection: {collection}")
        target = self.vectorstore_recipes

        meta = get_batch_meta(self.vectorstore_path, csv_path)
        start_offset = meta.get("rows_indexed", 0)

        total_rows = meta.get("total_rows")
        if total_rows is None:
            total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            meta["total_rows"] = total_rows

        total_batches = (total_rows + chunksize - 1) // chunksize
        rows_processed = start_offset
        batch_num = start_offset // chunksize

        logger.info(
            "Starting batched ingestion of %s (%d total rows, resuming from row %d)",
            path.name, total_rows, start_offset,
        )

        reader = pd.read_csv(csv_path, chunksize=chunksize)
        for chunk_df in reader:
            if rows_processed + len(chunk_df) <= start_offset:
                rows_processed += len(chunk_df)
                batch_num += 1
                continue

            docs = loader_fn(chunk_df)
            if docs:
                new_store = FAISS.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                if target is None:
                    target = new_store
                else:
                    target.merge_from(new_store)

                self.vectorstore_recipes = target
                self.vectorstore_path.mkdir(parents=True, exist_ok=True)
                target.save_local(str(self.vectorstore_path / "recipes_and_meals_db"))

            rows_processed += len(chunk_df)
            batch_num += 1

            meta["rows_indexed"] = rows_processed
            save_batch_meta(self.vectorstore_path, csv_path, meta)

            logger.info(
                "Batch %d/%d — %d/%d rows indexed",
                batch_num, total_batches, rows_processed, total_rows,
            )

        logger.info("Batched ingestion complete: %s (%d rows)", path.name, rows_processed)

    # ================================================================
    # CSV data loaders
    # ================================================================

    # ================================================================
    # Domain-specific public methods
    # ================================================================




def _to_float(value) -> float | None:
    """Safely coerce a JSON value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
