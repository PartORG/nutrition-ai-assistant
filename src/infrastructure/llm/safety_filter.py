"""
infrastructure.llm.safety_filter - Multi-provider recipe safety validation.

Implements SafetyFilterPort using a hybrid approach:
    1. Rule-based checks (avoid-lists, dietary restrictions, nutrition limits)
    2. LLM semantic check (catch subtle issues like "prosciutto is pork")

The LLM provider (openai / groq / ollama) is controlled by the
centralized LLM_PROVIDER setting.

Receives pre-parsed list[Recipe] objects from RecipeRAG (no LLM re-parsing needed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from domain.models import (
    UserIntent,
    NutritionConstraints,
    NutritionValues,
    Recipe,
    SafetyVerdict,
    SafetyIssue,
    RecipeSafetyResult,
    SafetyCheckResult,
)
from domain.exceptions import SafetyCheckError
from infrastructure.llm.llm_builder import build_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restriction-to-ingredient mapping for rule-based checks
# ---------------------------------------------------------------------------

RESTRICTION_INGREDIENT_MAP: dict[str, list[str]] = {
    "vegetarian": [
        "chicken", "beef", "pork", "lamb", "fish", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami", "anchovy",
        "shrimp", "crab", "lobster", "duck", "veal", "venison",
    ],
    "vegan": [
        "chicken", "beef", "pork", "lamb", "fish", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami", "anchovy",
        "shrimp", "crab", "lobster", "duck", "veal", "venison",
        "egg", "milk", "cheese", "butter", "cream", "yogurt", "honey",
        "whey", "casein", "ghee", "gelatin",
    ],
    "gluten-free": [
        "wheat", "flour", "bread", "pasta", "barley", "rye",
        "couscous", "semolina", "bulgur", "farro", "spelt",
    ],
    "lactose-free": [
        "milk", "cheese", "butter", "cream", "yogurt", "whey",
        "casein", "ghee", "ice cream",
    ],
    "pescatarian": [
        "chicken", "beef", "pork", "lamb", "meat", "turkey",
        "bacon", "ham", "sausage", "prosciutto", "salami",
        "duck", "veal", "venison",
    ],
}


class SafetyFilter:
    """Implements SafetyFilterPort using any supported LLM provider + rule-based checks."""

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434/",
        openai_api_key: str = "",
        groq_api_key: str = "",
        debug: bool = False,
    ):
        self._debug = debug
        self._llm = build_llm(
            provider=provider,
            model=model,
            temperature=0,
            json_mode=True,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
        )
        self._parser = JsonOutputParser()
        self._check_chain = self._build_check_chain()

    # ------------------------------------------------------------------
    # Public API (async)
    # ------------------------------------------------------------------

    async def check(
        self,
        recipes: list[Recipe],
        constraints: NutritionConstraints,
        intent: UserIntent,
    ) -> SafetyCheckResult:
        """Main entry: check pre-parsed recipes, return results."""
        loop = asyncio.get_event_loop()

        if not recipes:
            return SafetyCheckResult(
                safe_recipes_markdown="",
                summary="No recipes to check.",
            )

        if self._debug:
            logger.debug("Checking %d recipes", len(recipes))

        # Gather constraint data
        avoid_foods = constraints.avoid
        constraint_rules = constraints.constraints
        restrictions = intent.restrictions

        # Step 1 & 2: Rule-based checks
        rule_issues_per_recipe: list[list[SafetyIssue]] = []
        for recipe in recipes:
            issues: list[SafetyIssue] = []
            issues.extend(self._check_ingredients(recipe.ingredients, avoid_foods, restrictions))
            issues.extend(self._check_nutrition(recipe.nutrition, constraint_rules))
            rule_issues_per_recipe.append(issues)

        # Step 3: LLM semantic check
        llm_issues_map = await loop.run_in_executor(
            None, self._llm_semantic_check, recipes, constraints, intent,
        )

        # Step 4: Merge and determine verdicts
        recipe_verdicts: list[RecipeSafetyResult] = []
        safe_recipes: list[Recipe] = []

        for i, recipe in enumerate(recipes):
            all_issues = rule_issues_per_recipe[i] + llm_issues_map.get(recipe.name, [])

            if any(iss.severity in ("critical", "high") for iss in all_issues):
                verdict = SafetyVerdict.UNSAFE
            elif any(iss.severity == "medium" for iss in all_issues):
                verdict = SafetyVerdict.WARNING
            else:
                verdict = SafetyVerdict.SAFE

            rv = RecipeSafetyResult(
                recipe_name=recipe.name,
                verdict=verdict,
                issues=all_issues,
                recipe=recipe,
            )
            recipe_verdicts.append(rv)

            if verdict != SafetyVerdict.UNSAFE:
                safe_recipes.append(recipe)

            if self._debug:
                status = verdict.value.upper()
                logger.debug("  [%s] %s — %d issue(s)", status, recipe.name, len(all_issues))

        # Step 5: Build filtered markdown from safe Recipe objects
        safe_markdown = self._recipes_to_markdown(safe_recipes)

        # Build summary
        safe_count = len(safe_recipes)
        total_count = len(recipes)
        summary_parts = [f"{safe_count}/{total_count} recipes passed safety check."]
        for rv in recipe_verdicts:
            if rv.verdict == SafetyVerdict.UNSAFE:
                reasons = "; ".join(iss.description for iss in rv.issues[:3])
                summary_parts.append(f"  REJECTED '{rv.recipe_name}': {reasons}")
            elif rv.verdict == SafetyVerdict.WARNING:
                reasons = "; ".join(iss.description for iss in rv.issues[:3])
                summary_parts.append(f"  WARNING '{rv.recipe_name}': {reasons}")

        return SafetyCheckResult(
            recipe_verdicts=recipe_verdicts,
            safe_recipes_markdown=safe_markdown,
            summary="\n".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # LLM chain builders
    # ------------------------------------------------------------------

    def _build_check_chain(self):
        system = """You are a medical dietary safety checker. You ONLY flag CLEAR, DEFINITE safety violations. Do NOT flag minor or speculative concerns.

Your job is to catch issues that simple keyword matching would miss. For example:
- "prosciutto" is pork (relevant if pork is restricted)
- "ghee" is dairy (relevant if lactose-free)
- "soy sauce" contains gluten (relevant if gluten-free)

IMPORTANT RULES:
- Only flag an issue if an ingredient DIRECTLY and CLEARLY violates a stated constraint.
- Do NOT flag general health concerns (e.g., "oil is high in fat" for a diabetic patient).
- Do NOT flag LIMIT foods — those are advisory, not violations.
- Do NOT re-flag obvious ingredient matches (e.g., "peanut butter" matching "peanuts") — those are already handled by rules.
- When in doubt, do NOT flag it. Prefer false negatives over false positives.

For EACH recipe, check ONLY:
1. Hidden allergens/avoid foods that are not obvious from the ingredient name (category: "hidden_ingredient", severity: "high")
2. Ingredients that violate dietary restrictions in a non-obvious way (category: "restriction_violation", severity: "high")

Return JSON:
{{
  "recipe_checks": [
    {{
      "recipe_name": "name",
      "issues": [
        {{
          "category": "hidden_ingredient|restriction_violation",
          "severity": "high",
          "description": "human-readable explanation",
          "detail": "specific detail"
        }}
      ]
    }}
  ]
}}

If a recipe has NO issues, return an empty issues list for it. Most recipes should have NO issues.
Return ONLY valid JSON."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "PARSED RECIPES:\n{recipes_json}\n\nPATIENT CONSTRAINTS:\n- Restrictions/allergies: {restrictions}\n- Foods to AVOID: {avoid_foods}\n- Foods to LIMIT: {limit_foods}\n- Health conditions: {health_conditions}\n- Meal instructions: {instructions}\n- Nutrition limits: {nutrition_constraints}"),
        ])
        return prompt | self._llm | self._parser

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _word_match(word: str, text: str) -> bool:
        return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

    def _check_ingredients(
        self,
        ingredients: list[str],
        avoid_foods: list[str],
        restrictions: list[str],
    ) -> list[SafetyIssue]:
        issues: list[SafetyIssue] = []
        restrictions_lower = [r.lower() for r in restrictions]

        for ingredient in ingredients:
            ing_lower = ingredient.lower()

            for avoid in avoid_foods:
                if len(avoid.split()) > 2:
                    continue
                if self._word_match(avoid, ing_lower):
                    issues.append(SafetyIssue(
                        category="avoid_food",
                        severity="critical",
                        description=f"Ingredient '{ingredient}' matches avoid item '{avoid}'",
                        detail=f"avoid_match: '{avoid}' in '{ingredient}'",
                    ))

            for restriction in restrictions_lower:
                banned_items = RESTRICTION_INGREDIENT_MAP.get(restriction, [])
                for banned in banned_items:
                    if self._word_match(banned, ing_lower):
                        issues.append(SafetyIssue(
                            category="restriction_violation",
                            severity="critical",
                            description=f"Ingredient '{ingredient}' violates '{restriction}' restriction",
                            detail=f"banned_ingredient: '{banned}'",
                        ))
                        break

        return issues

    def _check_nutrition(
        self,
        nutrition: NutritionValues,
        constraint_rules: dict[str, dict[str, float | None]],
    ) -> list[SafetyIssue]:
        issues: list[SafetyIssue] = []
        field_map: dict[str, float | None] = {
            "sugar_g": nutrition.sugar_g,
            "sodium_mg": nutrition.sodium_mg,
            "fiber_g": nutrition.fiber_g,
            "protein_g": nutrition.protein_g,
            "saturated_fat_g": nutrition.saturated_fat_g,
            "calories": nutrition.calories,
            "carbs_g": nutrition.carbs_g,
            "fat_g": nutrition.fat_g,
        }

        for nutrient, rule in constraint_rules.items():
            value = field_map.get(nutrient)
            if value is None:
                continue
            max_val = rule.get("max")
            min_val = rule.get("min")
            if max_val is not None and value > max_val:
                issues.append(SafetyIssue(
                    category="nutrition_limit",
                    severity="medium",
                    description=f"{nutrient} ({value}) exceeds maximum ({max_val})",
                    detail=f"{nutrient}: {value} > max {max_val}",
                ))
            if min_val is not None and value < min_val:
                issues.append(SafetyIssue(
                    category="nutrition_limit",
                    severity="medium",
                    description=f"{nutrient} ({value}) below minimum ({min_val})",
                    detail=f"{nutrient}: {value} < min {min_val}",
                ))

        return issues

    def _llm_semantic_check(
        self,
        recipes: list[Recipe],
        constraints: NutritionConstraints,
        intent: UserIntent,
    ) -> dict[str, list[SafetyIssue]]:
        recipes_for_llm = []
        for r in recipes:
            recipes_for_llm.append({
                "name": r.name,
                "ingredients": r.ingredients,
                "nutrition": {
                    "calories": r.nutrition.calories,
                    "protein_g": r.nutrition.protein_g,
                    "carbs_g": r.nutrition.carbs_g,
                    "fat_g": r.nutrition.fat_g,
                    "sugar_g": r.nutrition.sugar_g,
                    "sodium_mg": r.nutrition.sodium_mg,
                },
            })

        nutrition_str = json.dumps(constraints.constraints) if constraints.constraints else "None"

        try:
            result = self._check_chain.invoke({
                "recipes_json": json.dumps(recipes_for_llm, indent=2),
                "restrictions": ", ".join(intent.restrictions) or "None",
                "avoid_foods": ", ".join(constraints.avoid) or "None",
                "limit_foods": ", ".join(constraints.limit) or "None",
                "health_conditions": ", ".join(intent.health_conditions) or "None",
                "instructions": ", ".join(intent.instructions) or "None",
                "nutrition_constraints": nutrition_str,
            })

            issues_map: dict[str, list[SafetyIssue]] = {}
            for check in result.get("recipe_checks", []):
                name = check.get("recipe_name", "")
                issues: list[SafetyIssue] = []
                for iss in check.get("issues", []):
                    issues.append(SafetyIssue(
                        category=iss.get("category", "unknown"),
                        severity=iss.get("severity", "medium"),
                        description=iss.get("description", ""),
                        detail=iss.get("detail", ""),
                    ))
                if issues:
                    issues_map[name] = issues
            return issues_map
        except Exception as e:
            if self._debug:
                logger.debug("LLM semantic check error: %s", e)
            return {}

    @staticmethod
    def _recipes_to_markdown(recipes: list[Recipe]) -> str:
        """Build display markdown from a list of Recipe objects."""
        if not recipes:
            return ""

        parts: list[str] = []
        for i, r in enumerate(recipes, 1):
            lines: list[str] = [f"## {i}. {r.name}"]
            if r.why_recommended:
                lines.append(f"*{r.why_recommended}*")
            if r.servings:
                lines.append(f"**Servings:** {r.servings}")
            if r.prep_time:
                lines.append(f"**Prep time:** {r.prep_time}")
            if r.ingredients:
                lines.append("\n**Ingredients:**")
                for ing in r.ingredients:
                    lines.append(f"- {ing}")
            if r.cook_instructions:
                lines.append(f"\n**Instructions:**\n{r.cook_instructions}")
            n = r.nutrition
            if n and any(v is not None for v in [n.calories, n.protein_g, n.carbs_g, n.fat_g]):
                nutr_parts: list[str] = []
                if n.calories is not None:
                    nutr_parts.append(f"Calories: {n.calories:.0f} kcal")
                if n.protein_g is not None:
                    nutr_parts.append(f"Protein: {n.protein_g:.1f}g")
                if n.carbs_g is not None:
                    nutr_parts.append(f"Carbs: {n.carbs_g:.1f}g")
                if n.fat_g is not None:
                    nutr_parts.append(f"Fat: {n.fat_g:.1f}g")
                if n.fiber_g is not None:
                    nutr_parts.append(f"Fiber: {n.fiber_g:.1f}g")
                if n.sodium_mg is not None:
                    nutr_parts.append(f"Sodium: {n.sodium_mg:.0f}mg")
                lines.append("\n**Nutrition (per serving):** " + " | ".join(nutr_parts))
            parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts)


# Backward-compatible alias (so old imports still work)
OllamaSafetyFilter = SafetyFilter
