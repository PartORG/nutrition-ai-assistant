"""
Tests for the structured-output conversion.

Before this, three LLM paths parsed strings: recipes via a greedy regex,
medical constraints via fence-stripping plus a regex, and safety verdicts via
JsonOutputParser feeding a chain of `.get()` defaults. All three now bind a
Pydantic schema through the provider's structured-output API.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from infrastructure.llm.safety_filter import _normalize_severity
from infrastructure.rag.base_rag import BaseRAG
from infrastructure.rag.medical_rag import (
    MedicalRAG,
    _ConstraintsSchema,
    _NutrientLimit,
)
from infrastructure.rag.recipe_rag import (
    RecipeNutritionRAG,
    _first_json_object,
    _NutritionSchema,
    _RecipeListSchema,
    _RecipeSchema,
)

# ---------------------------------------------------------------------------
# Balanced-brace extraction (fallback path)
# ---------------------------------------------------------------------------

def test_extracts_first_complete_object():
    assert _first_json_object('{"a": 1} trailing') == '{"a": 1}'


def test_stops_at_first_object_not_the_last_brace():
    assert _first_json_object('{"a": 1} junk {"b": 2}') == '{"a": 1}'


def test_handles_nesting():
    assert _first_json_object('noise {"a": {"b": [1, 2]}} more') == '{"a": {"b": [1, 2]}}'


def test_ignores_braces_inside_strings():
    assert _first_json_object('{"a": "}{"}') == '{"a": "}{"}'


def test_ignores_escaped_quotes():
    assert _first_json_object(r'{"a": "say \"hi\""}') == r'{"a": "say \"hi\""}'


def test_truncated_object_is_unrecoverable():
    assert _first_json_object('{"a": {"b": 1}') is None


def test_no_object_returns_none():
    assert _first_json_object("sorry, I cannot help") is None
    assert _first_json_object("") is None


# ---------------------------------------------------------------------------
# Recipe schema -> domain
# ---------------------------------------------------------------------------

def test_schema_maps_to_domain_recipes():
    parsed = _RecipeListSchema(
        recipes=[
            _RecipeSchema(
                name="  Paneer Skillet  ",
                ingredients=["200g paneer", "240g chickpeas"],
                cook_instructions="1. Heat oil.",
                servings=2,
                nutrition=_NutritionSchema(calories=470, protein_g=38),
            )
        ]
    )
    [recipe] = RecipeNutritionRAG._schema_to_recipes(parsed)

    assert recipe.name == "Paneer Skillet"
    assert recipe.ingredients == ["200g paneer", "240g chickpeas"]
    assert recipe.nutrition.calories == 470
    assert recipe.nutrition.protein_g == 38
    assert recipe.servings == 2


def test_zero_servings_falls_back_to_one():
    parsed = _RecipeListSchema(recipes=[_RecipeSchema(name="X", servings=0)])
    assert RecipeNutritionRAG._schema_to_recipes(parsed)[0].servings == 1


def test_empty_schema_maps_to_empty_list():
    assert RecipeNutritionRAG._schema_to_recipes(_RecipeListSchema()) == []


# ---------------------------------------------------------------------------
# Medical constraints schema -> domain
# ---------------------------------------------------------------------------

def test_constraints_flatten_to_the_domain_dict_shape():
    schema = _ConstraintsSchema(
        avoid=["pork", "shellfish"],
        limit=["salt"],
        sodium_mg=_NutrientLimit(max=1500),
        fiber_g=_NutrientLimit(min=25),
        notes="Low sodium diet.",
    )
    domain = schema.to_domain()

    assert domain.avoid == ["pork", "shellfish"]
    assert domain.limit == ["salt"]
    assert domain.constraints == {"sodium_mg": {"max": 1500.0}, "fiber_g": {"min": 25.0}}
    assert domain.notes == "Low sodium diet."


def test_absent_limits_are_omitted_not_nulled():
    """A nutrient with no bound must not appear as {'max': None}."""
    domain = _ConstraintsSchema(sugar_g=_NutrientLimit()).to_domain()
    assert domain.constraints == {}


# ---------------------------------------------------------------------------
# Safety severity
# ---------------------------------------------------------------------------

def test_known_severities_pass_through():
    for severity in ("critical", "high", "medium"):
        assert _normalize_severity(severity) == severity


def test_severity_is_case_and_space_insensitive():
    assert _normalize_severity("  CRITICAL ") == "critical"


def test_unknown_severity_escalates_rather_than_downgrades():
    """Defaulting to 'medium' turned an unrecognised critical finding into a warning."""
    assert _normalize_severity("catastrophic") == "high"
    assert _normalize_severity("") == "high"
    assert _normalize_severity(None) == "high"


# ---------------------------------------------------------------------------
# Chain wiring
# ---------------------------------------------------------------------------

class _SchemaRAG(BaseRAG):
    SYSTEM_PROMPT = "{input} {context}"
    OUTPUT_SCHEMA = _RecipeListSchema

    def _ingest_documents(self):
        return []

    def _ingest_single_file(self, file_path):
        return []


class _PlainRAG(_SchemaRAG):
    OUTPUT_SCHEMA = None


def _built_chain(cls, tmp_path):
    rag = cls(vectorstore_path=str(tmp_path))
    rag.llm = FakeListChatModel(responses=["{}"])
    rag.retriever = lambda _q: []
    rag._build_chain()
    return rag.rag_chain


def test_plain_rag_builds_a_string_chain(tmp_path):
    """RAGs without a schema keep the original string contract."""
    assert _built_chain(_PlainRAG, tmp_path) is not None


def test_provider_without_structured_output_degrades_instead_of_crashing(tmp_path, caplog):
    """A provider lacking with_structured_output must not break startup.

    FakeListChatModel raises NotImplementedError, standing in for any provider
    that cannot bind a schema. The chain must still build so the subclass's
    string-parsing fallback can take over.
    """
    with caplog.at_level(logging.WARNING):
        chain = _built_chain(_SchemaRAG, tmp_path)

    assert chain is not None, "startup must survive a provider without schema support"
    assert "does not support structured output" in caplog.text


def test_schema_rags_get_a_chat_model_on_ollama(tmp_path):
    """Regression: OllamaLLM cannot bind a schema; only ChatOllama can.

    build_llm returns OllamaLLM unless chat_model=True. Without that flag every
    OUTPUT_SCHEMA would be silently dropped on Ollama — the default provider —
    and responses parsed as strings, defeating the whole conversion.
    """
    for cls, kwargs in (
        (RecipeNutritionRAG, {"data_folder": "data"}),
        (MedicalRAG, {"folder_paths": ["pdfs"]}),
    ):
        rag = cls(vectorstore_path=str(tmp_path), llm_provider="ollama", **kwargs)
        llm = rag._build_llm()

        assert type(llm).__name__ == "ChatOllama", (
            f"{cls.__name__} built {type(llm).__name__}, which cannot bind a schema"
        )
        llm.with_structured_output(cls.OUTPUT_SCHEMA)  # must not raise


def test_schema_rag_binds_structured_output_when_supported(tmp_path):
    """When the provider does support it, the schema reaches the provider."""
    bound = {}

    class _CapableLLM(FakeListChatModel):
        def with_structured_output(self, schema, **kwargs):
            bound["schema"] = schema
            return self

    rag = _SchemaRAG(vectorstore_path=str(tmp_path))
    rag.llm = _CapableLLM(responses=["{}"])
    rag.retriever = lambda _q: []
    rag._build_chain()

    assert bound["schema"] is _RecipeListSchema
