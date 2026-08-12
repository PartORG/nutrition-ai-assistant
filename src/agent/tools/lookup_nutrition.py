"""
agent.tools.lookup_nutrition - Food composition lookup from the database.

Before this tool existed, the agent prompt routed "how much protein is in
cornstarch" to the model's own memory ("answer DIRECTLY, no tool needed") while
a 36 MB FAISS index of the same data sat unused. Now the answer comes from the
nutrition_facts table, so it is exact rather than recalled.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolResult
from application.context import SessionContext
from domain.models import NutritionFact
from domain.ports import NutritionFactsRepository

logger = logging.getLogger(__name__)

# Nutrients worth showing by default. The table carries all 74, but a wall of
# amino acids helps nobody — `nutrient` narrows to a specific one on request.
_SUMMARY_NUTRIENTS: list[tuple[str, str]] = [
    ("calories", "Calories"),
    ("protein", "Protein"),
    ("carbohydrate", "Carbs"),
    ("fat", "Fat"),
    ("fiber", "Fiber"),
    ("sugars", "Sugars"),
    ("sodium", "Sodium"),
    ("saturated_fat", "Saturated fat"),
]

# What users say -> what the column is called. The dataset carries two known
# typos ("irom", "zink"), which are preserved in the schema and mapped here.
_NUTRIENT_ALIASES: dict[str, str] = {
    "iron": "irom",
    "zinc": "zink",
    "carbs": "carbohydrate",
    "carbohydrates": "carbohydrate",
    "sugar": "sugars",
    "vitamin c": "vitamin_c",
    "vitamin d": "vitamin_d",
    "vitamin a": "vitamin_a",
    "vitamin e": "vitamin_e",
    "vitamin k": "vitamin_k",
    "vitamin b12": "vitamin_b12",
    "vitamin b6": "vitamin_b6",
    "saturated fat": "saturated_fat",
    "kcal": "calories",
    "energy": "calories",
}


def normalize_nutrient(name: str) -> str:
    """Map a user's wording onto a column name."""
    key = (name or "").strip().lower()
    if key in _NUTRIENT_ALIASES:
        return _NUTRIENT_ALIASES[key]
    return key.replace(" ", "_").replace("-", "_")


class LookupNutritionInput(BaseModel):
    """Input schema for the lookup_nutrition tool."""

    food: str = Field(
        default="",
        description="Food to look up, e.g. 'cornstarch', 'chicken breast'.",
    )
    nutrient: str = Field(
        default="",
        description=(
            "Optional single nutrient, e.g. 'iron', 'vitamin c', 'protein'. "
            "With `food`, returns just that nutrient. Without `food`, returns "
            "the foods highest in it."
        ),
    )


class LookupNutritionTool(BaseTool):
    """Look up reference nutrition values for a food."""

    name = "lookup_nutrition"
    description = (
        "Look up the nutritional content of a FOOD or INGREDIENT from the "
        "nutrition database. Use for: 'how much protein is in cornstarch', "
        "'calories in an avocado', 'is spinach high in iron', "
        "'what foods are highest in vitamin C', 'nutrition facts for salmon'. "
        "This is reference data about foods themselves — for what the USER has "
        "eaten today, use nutrition_status instead."
    )

    def __init__(self, facts_repo: NutritionFactsRepository):
        self._repo = facts_repo

    def get_schema(self) -> type[BaseModel]:
        return LookupNutritionInput

    async def execute(
        self,
        ctx: SessionContext,
        food: str = "",
        nutrient: str = "",
        **kwargs,
    ) -> ToolResult:
        food = (food or "").strip()
        nutrient_col = normalize_nutrient(nutrient) if nutrient else ""

        if not food and not nutrient_col:
            return ToolResult(
                output="Tell me which food to look up, e.g. 'nutrition in cornstarch'."
            )

        try:
            if not food:
                return await self._top_foods(nutrient, nutrient_col)
            return await self._food_facts(food, nutrient, nutrient_col)
        except ValueError as e:
            logger.info("lookup_nutrition rejected input: %s", e)
            return ToolResult(output=f"I don't have data for '{nutrient}'.")
        except Exception:
            logger.exception("lookup_nutrition failed")
            return ToolResult(output="The nutrition database is unavailable right now.")

    # ------------------------------------------------------------------

    async def _top_foods(self, requested: str, column: str) -> ToolResult:
        facts = await self._repo.top_by_nutrient(column, limit=10)
        if not facts:
            return ToolResult(output=f"No foods found with data for {requested}.")

        lines = [f"**Foods highest in {requested}** (per 100 g):", ""]
        lines += [
            f"{i}. {f.name} — {f.format_amount(column)}"
            for i, f in enumerate(facts, 1)
        ]
        return ToolResult(output="\n".join(lines), data=facts)

    async def _food_facts(self, food: str, requested: str, column: str) -> ToolResult:
        # Name matching is lexical, so "salmon" can rank "Salmonberries, raw"
        # first. Reporting a berry's macros as salmon's is the kind of wrong
        # that matters here, so an inexact match always names what it matched
        # and offers the alternatives rather than answering silently.
        matches = await self._repo.search(food, limit=4)
        if not matches:
            return ToolResult(
                output=(
                    f"'{food}' is not in the nutrition database. It matches food "
                    f"names literally, so try a different wording."
                )
            )

        fact = matches[0]
        exact = fact.name.strip().lower() == food.strip().lower()
        alternatives = [m.name for m in matches[1:4]]

        if column:
            if fact.get(column) is None:
                body = f"The database has no {requested} value for **{fact.name}**."
            else:
                body = (
                    f"**{fact.name}** — {requested}: "
                    f"{fact.format_amount(column)} per {fact.serving_size or '100 g'}"
                )
        else:
            body = self._summarize(fact)

        if not exact:
            note = f"Closest match for '{food}':"
            if alternatives:
                note += f" (others: {', '.join(alternatives)})"
            body = f"{note}\n\n{body}"

        return ToolResult(output=body, data=fact)

    @staticmethod
    def _summarize(fact: NutritionFact) -> str:
        lines = [f"**{fact.name}** (per {fact.serving_size or '100 g'})", ""]
        for column, label in _SUMMARY_NUTRIENTS:
            if fact.get(column) is not None:
                lines.append(f"- {label}: {fact.format_amount(column)}")
        if len(lines) == 2:
            return f"**{fact.name}** is in the database but has no macro values recorded."
        return "\n".join(lines)
