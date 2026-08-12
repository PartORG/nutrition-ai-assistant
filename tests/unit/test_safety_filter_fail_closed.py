"""
Regression guard: the safety filter must never report an unverified recipe as SAFE.

Before this test existed, `_llm_semantic_check` caught every exception and
returned `{}`, which the caller merged as "no issues found" — a failed or
timed-out semantic check silently produced a SAFE verdict for a user with
allergies or medical conditions.
"""

from __future__ import annotations

import pytest

from domain.exceptions import SafetyCheckError
from domain.models import (
    NutritionConstraints,
    NutritionValues,
    Recipe,
    SafetyVerdict,
    UserIntent,
)
from infrastructure.llm.safety_filter import SafetyFilter


def _filter_with_broken_semantic_check() -> SafetyFilter:
    """A SafetyFilter whose LLM call always blows up, with no LLM constructed."""
    f = SafetyFilter.__new__(SafetyFilter)  # bypass __init__ so no Ollama is needed
    f._debug = False

    def _boom(*_args, **_kwargs):
        # Matches the real contract: _llm_semantic_check raises SafetyCheckError.
        raise SafetyCheckError("ollama unreachable")

    f._llm_semantic_check = _boom
    return f


def _recipe() -> Recipe:
    return Recipe(
        name="Prosciutto Pasta",
        ingredients=["pasta", "prosciutto", "olive oil"],
        cook_instructions="Boil pasta. Add prosciutto.",
        nutrition=NutritionValues(calories=600, protein_g=25, carbs_g=70, fat_g=20),
        servings=2,
    )


async def test_semantic_failure_does_not_yield_safe_verdict():
    """With restrictions in play, a failed semantic check must degrade to WARNING."""
    result = await _filter_with_broken_semantic_check().check(
        recipes=[_recipe()],
        constraints=NutritionConstraints(avoid=["pork"]),
        intent=UserIntent(restrictions=["vegetarian"]),
    )

    verdicts = [rv.verdict for rv in result.recipe_verdicts]
    assert SafetyVerdict.SAFE not in verdicts, (
        "safety filter reported SAFE while the semantic check was failing"
    )
    assert "INCOMPLETE" in result.summary


async def test_semantic_failure_is_harmless_with_nothing_to_check():
    """No restrictions means the semantic layer had nothing to catch — stay SAFE."""
    result = await _filter_with_broken_semantic_check().check(
        recipes=[_recipe()],
        constraints=NutritionConstraints(),
        intent=UserIntent(),
    )

    assert [rv.verdict for rv in result.recipe_verdicts] == [SafetyVerdict.SAFE]
    assert "INCOMPLETE" not in result.summary


def test_semantic_check_raises_rather_than_returning_empty():
    """The low-level helper must surface failure, not return a falsy 'all clear'."""
    f = SafetyFilter.__new__(SafetyFilter)
    f._debug = False

    class _ExplodingChain:
        def invoke(self, _payload):
            raise RuntimeError("malformed JSON from model")

    f._check_chain = _ExplodingChain()

    with pytest.raises(SafetyCheckError):
        f._llm_semantic_check([_recipe()], NutritionConstraints(), UserIntent())
