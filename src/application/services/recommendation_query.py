"""
application.services.recommendation_query - Building the RAG query, and the
arithmetic and salvage helpers around it.

Extracted from RecommendationService, which mixed four jobs: orchestrating the
pipeline, constructing a 136-line prompt, doing per-meal nutrition arithmetic,
and salvaging malformed rows written by older versions of the app. Only the
first is orchestration; the rest are pure functions that never touched `self`.
"""

from __future__ import annotations

import json
import logging

from domain.models import NutritionConstraints, UserIntent

logger = logging.getLogger(__name__)


def build_augmented_query(
    original_query: str,
    intent: UserIntent,
    constraints: NutritionConstraints,
    daily_totals: dict | None = None,
) -> str:
    """Build an enriched query for the Recipe/Nutrition RAG.

    Design principles:
    1. RETRIEVAL-FIRST — the query is used by SmartRetriever for
       similarity search, so food/recipe keywords must appear early
       and prominently to pull the right documents.
    2. COMPACT — the RAG system prompt already instructs the LLM on
       output format; this query should only provide *what* to cook,
       not *how* to answer.
    3. NO DUPLICATION — each piece of info appears once.
    4. STRUCTURED FOR LLM — the {input} placeholder in the RAG prompt
       receives this text verbatim, so it should be scannable.
    """
    # --- Part 1: Recipe search core (drives retriever) ---------------
    # Put the natural-language request first — this is what the vector
    # search primarily matches against.
    parts: list[str] = [original_query]

    # Add preferred ingredients / cuisine — these are high-signal tokens
    # for the recipe vectorstore.
    if intent.preferences:
        parts.append(
            "Preferred ingredients: " + ", ".join(intent.preferences)
        )

    if intent.instructions:
        parts.append(
            "Special instructions: " + ", ".join(intent.instructions)
        )

    # --- Part 2: Constraints (compact, for LLM context) -------------
    constraint_lines: list[str] = []

    if intent.health_conditions:
        constraint_lines.append(
            "Medical conditions: " + ", ".join(intent.health_conditions)
        )

    if intent.restrictions:
        constraint_lines.append(
            "Dietary restrictions: " + ", ".join(intent.restrictions)
        )

    # Free-text medical advice from the DB / Medical RAG (e.g. "should eat
    # frequent small meals", "avoid high-GI foods").  Skip the generic
    # fallback note that carries no real information.
    _GENERIC_NOTE = "No specific medical conditions provided"
    if constraints.notes and constraints.notes.strip() != _GENERIC_NOTE:
        constraint_lines.append("Medical advice: " + constraints.notes.strip())

    # Numeric limits — only include non-empty ones
    rules = constraints.constraints
    limit_tokens: list[str] = []
    for nutrient, label, unit in [
        ("sugar_g", "sugar", "g"),
        ("sodium_mg", "sodium", "mg"),
        ("fiber_g", "fiber", "g"),
        ("protein_g", "protein", "g"),
        ("calories", "calories", "kcal"),
    ]:
        rule = rules.get(nutrient) or {}
        if rule.get("max") is not None:
            limit_tokens.append(f"{label} max {rule['max']}{unit}")
        if rule.get("min") is not None:
            limit_tokens.append(f"{label} min {rule['min']}{unit}")

    if limit_tokens:
        constraint_lines.append(
            "Nutrient limits: " + ", ".join(limit_tokens)
        )

    if constraints.avoid:
        constraint_lines.append(
            "Avoid: " + ", ".join(constraints.avoid)
        )

    if constraints.limit:
        constraint_lines.append(
            "Limit: " + ", ".join(constraints.limit)
        )

    if constraint_lines:
        parts.append("\n".join(constraint_lines))

    # --- Part 3: Daily budget (remaining allowance) ------------------
    # If the user has already eaten some meals today, show what has been
    # consumed and what's left. This helps the Recipe RAG propose meals
    # that fit within the remaining daily limits.
    if daily_totals:
        budget_lines: list[str] = []
        consumed_tokens: list[str] = []
        remaining_tokens: list[str] = []
        rules = constraints.constraints  # already reduced by adjust_for_daily_budget

        for nutrient, label, unit in [
            ("calories", "calories", "kcal"),
            ("sugar_g", "sugar", "g"),
            ("sodium_mg", "sodium", "mg"),
            ("fiber_g", "fiber", "g"),
            ("protein_g", "protein", "g"),
        ]:
            consumed = daily_totals.get(nutrient, 0.0)
            if consumed <= 0:
                continue
            consumed_tokens.append(f"{label} {consumed:.0f}{unit}")

            rule = rules.get(nutrient) or {}
            max_val = rule.get("max")
            if max_val is not None:
                remaining_tokens.append(f"{label} {max_val:.0f}{unit} remaining")

        if consumed_tokens:
            budget_lines.append("Already consumed today: " + ", ".join(consumed_tokens))
        if remaining_tokens:
            budget_lines.append("Remaining daily budget: " + ", ".join(remaining_tokens))
        if budget_lines:
            budget_lines.insert(
                0,
                "IMPORTANT — the nutrient limits below reflect the user's "
                "REMAINING daily allowance after meals already eaten today:",
            )
            parts.append("\n".join(budget_lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def adjust_for_daily_budget(
    constraints: NutritionConstraints,
    daily_totals: dict[str, float],
) -> NutritionConstraints:
    """Return a new NutritionConstraints where every max limit is reduced
    by the amount the user has already consumed today.

    The medical_advice table stores DAILY limits (e.g. sugar_g: 50 g/day).
    If the user already ate a meal with 20 g of sugar, only 30 g remain for
    new meals.  Clamped to 0 so the remaining budget is never negative.

    daily_totals keys match constraint keys: 'sugar_g', 'sodium_mg',
    'fiber_g', 'protein_g', 'calories', 'carbs_g', 'fat_g'.
    """
    if not daily_totals or not constraints.constraints:
        return constraints

    new_rules: dict = {}
    changed = False
    for nutrient, rule in constraints.constraints.items():
        consumed = daily_totals.get(nutrient, 0.0)
        max_val = rule.get("max")
        min_val = rule.get("min")

        if max_val is not None and consumed > 0:
            remaining = max(0.0, max_val - consumed)
            new_rules[nutrient] = {"max": round(remaining, 2), "min": min_val}
            changed = True
        else:
            new_rules[nutrient] = rule

    if not changed:
        return constraints

    return NutritionConstraints(
        dietary_goals=constraints.dietary_goals,
        foods_to_increase=constraints.foods_to_increase,
        avoid=constraints.avoid,
        limit=constraints.limit,
        constraints=new_rules,
        notes=constraints.notes,
    )


def split_or_empty(value: str | None) -> list[str]:
    """Split a comma/newline-separated string into a list, or return []."""
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def parse_constraints_str(value: str | None) -> dict:
    """Parse a JSON-like constraints string, or return empty dict."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def try_parse_corrupt_advice(raw: str) -> NutritionConstraints | None:
    """Try to parse a corrupt medical_advice DB value into NutritionConstraints.

    Handles two legacy storage formats:
    - Markdown-fenced JSON:  ```json\\n{...}\\n```
    - Raw JSON blob stored directly in the medical_advice column

    Returns None if the text cannot be parsed.
    """
    import re

    text = raw.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Last resort: find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(data, dict):
        return None

    goals = data.get("dietary_goals", [])
    if isinstance(goals, str):
        goals = [goals]

    return NutritionConstraints(
        dietary_goals=goals,
        foods_to_increase=data.get("foods_to_increase", []),
        avoid=data.get("avoid", []),
        limit=data.get("limit", []),
        constraints=data.get("constraints", {}),
        notes=data.get("notes", ""),
    )
