"""
agent.tools.nutrition_status - Daily nutrition tracking status.

Queries today's saved meals from the database and reports consumed
vs. remaining nutrition relative to the user's medical constraints.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from domain.ports import NutritionRepository, MedicalRepository
from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# (nutrition_history attr, constraint dict key, display label, unit)
_NUTRIENT_LABELS: list[tuple[str, str, str, str]] = [
    ("calories",      "calories",  "Calories", "kcal"),
    ("protein",       "protein_g", "Protein",  "g"),
    ("carbohydrates", "carbs_g",   "Carbs",    "g"),
    ("fat",           "fat_g",     "Fat",      "g"),
    ("fiber",         "fiber_g",   "Fiber",    "g"),
    ("sugar",         "sugar_g",   "Sugar",    "g"),
    ("sodium",        "sodium_mg", "Sodium",   "mg"),
]


class NutritionStatusInput(BaseModel):
    """Input schema for the nutrition_status tool."""

    question: str = Field(
        description="The user's question about their daily nutrition status"
    )


class NutritionStatusTool(BaseTool):
    """Report today's consumed nutrition vs daily limits from the database."""

    name = "nutrition_status"
    description = (
        "Look up the user's nutrition intake for today from the database. "
        "Use when user asks: 'Have I eaten enough today?', "
        "'How many calories have I consumed?', 'How much protein is left?', "
        "'What is my calorie budget for today?', 'How much have I eaten?', "
        "or any question about their current daily nutrition progress or status."
    )

    def __init__(
        self,
        nutrition_repo: NutritionRepository,
        medical_repo: MedicalRepository,
    ):
        self._nutrition_repo = nutrition_repo
        self._medical_repo = medical_repo

    def get_schema(self) -> type[BaseModel]:
        return NutritionStatusInput

    async def execute(
        self,
        ctx: SessionContext,
        question: str = "",
        **kwargs,
    ) -> ToolResult:
        """Retrieve today's nutrition totals and compare against daily limits."""
        try:
            records = await self._nutrition_repo.get_today_by_user(ctx.user_id)
        except Exception:
            logger.exception(
                "Failed to load today's nutrition for user %d", ctx.user_id
            )
            return ToolResult(
                output=(
                    "I couldn't retrieve your nutrition data right now. "
                    "Please try again in a moment."
                )
            )

        # Sum consumed nutrients from today's saved meals
        totals: dict[str, float] = {
            "calories": 0.0,
            "protein": 0.0,
            "carbohydrates": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 0.0,
        }
        for r in records:
            for key in totals:
                totals[key] += getattr(r, key, 0.0) or 0.0

        meal_count = len(records)

        # Try to load daily limits from the user's medical advice record
        limits: dict[str, dict] = {}
        try:
            advice_list = await self._medical_repo.get_by_user(ctx.user_id)
            if advice_list and advice_list[0].dietary_constraints:
                raw = advice_list[0].dietary_constraints
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, dict):
                    limits = parsed
        except Exception:
            logger.debug(
                "Could not load medical constraints for user %d", ctx.user_id
            )

        return ToolResult(output=_format_status(totals, limits, meal_count))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_status(
    totals: dict[str, float],
    limits: dict[str, dict],
    meal_count: int,
) -> str:
    """Render a markdown summary of today's consumed nutrition."""
    lines: list[str] = ["### Today's Nutrition Summary\n"]
    lines.append(f"**Meals logged today:** {meal_count}")

    any_consumed = any(v > 0 for v in totals.values())
    if not any_consumed:
        lines.append("\nNo meals have been saved yet today.")
        lines.append(
            "Ask me to suggest recipes and save them to start tracking your nutrition!"
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("| Nutrient | Consumed | Daily Limit | Remaining |")
    lines.append("|----------|----------|-------------|-----------|")

    for attr, constraint_key, label, unit in _NUTRIENT_LABELS:
        consumed = totals.get(attr, 0.0)
        rule = limits.get(constraint_key) or {}
        max_val: Optional[float] = rule.get("max")

        # Skip rows that are zero and have no configured limit
        if consumed <= 0 and max_val is None:
            continue

        consumed_str = f"{consumed:.0f} {unit}"
        limit_str = f"{max_val:.0f} {unit}" if max_val is not None else "—"

        if max_val is not None:
            remaining = max(0.0, max_val - consumed)
            remaining_str = f"{remaining:.0f} {unit}"
            if consumed >= max_val:
                remaining_str += " ⚠️ over limit"
            elif remaining < max_val * 0.1:
                remaining_str += " 🔶 almost full"
        else:
            remaining_str = "—"

        lines.append(f"| {label} | {consumed_str} | {limit_str} | {remaining_str} |")

    if limits:
        lines.append("\n*Limits are based on your medical profile.*")
    else:
        lines.append(
            "\n*No daily limits found. Add health conditions in your profile "
            "to get personalised limits.*"
        )

    return "\n".join(lines)
