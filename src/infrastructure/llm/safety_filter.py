"""
infrastructure.llm.safety_filter - Multi-provider recipe safety validation.

Implements SafetyFilterPort using a hybrid approach:
    1. Rule-based checks (avoid-lists, dietary restrictions, nutrition limits)
    2. LLM semantic check (catch subtle issues like "prosciutto is pork")

Receives pre-parsed list[Recipe] objects from RecipeRAG (no LLM re-parsing needed).
The LLM provider (openai / groq / ollama) is controlled by the
centralized LLM_PROVIDER setting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from domain.exceptions import SafetyCheckError
from domain.models import (
    NutritionConstraints,
    NutritionValues,
    Recipe,
    RecipeSafetyResult,
    SafetyCheckResult,
    SafetyIssue,
    SafetyVerdict,
    UserIntent,
)
from infrastructure.llm.llm_builder import build_llm
from infrastructure.llm.safety_rules import (
    RESTRICTION_INGREDIENT_MAP,
    SAFETY_CHECK_PROMPT,
)

logger = logging.getLogger(__name__)

_VALID_SEVERITIES = {"critical", "high", "medium"}

# Words carrying no food identity, dropped when splitting a multi-word avoid
# item so that "raw shellfish products" keys on "shellfish", not "products".
_FILLER_WORDS = {
    "raw", "cooked", "fresh", "dried", "frozen", "canned", "products", "product",
    "food", "foods", "items", "item", "based", "free", "with", "without", "and",
    "any", "all", "the", "whole",
}


def _normalize_severity(value: str) -> str:
    """Map a model's severity onto the three the verdict logic understands.

    Defaulting an unrecognised value to "medium" downgrades it to a warning.
    Anything unfamiliar is treated as "high" instead, so an unexpected label
    cannot quietly turn a rejection into a pass.
    """
    severity = (value or "").strip().lower()
    if severity in _VALID_SEVERITIES:
        return severity
    logger.warning("Unrecognised severity %r from model — treating as high", value)
    return "high"


# ---------------------------------------------------------------------------



class _IssueSchema(BaseModel):
    """One hidden-ingredient finding."""

    category: str = Field(
        default="hidden_ingredient",
        description="'hidden_ingredient' or 'restriction_violation'",
    )
    severity: str = Field(
        default="medium", description="'critical', 'high' or 'medium'"
    )
    description: str = Field(
        default="", description="Short explanation naming the restriction breached"
    )
    detail: str = Field(default="")


class _RecipeCheckSchema(BaseModel):
    recipe_name: str = Field(default="")
    issues: list[_IssueSchema] = Field(default_factory=list)


class _SafetyCheckSchema(BaseModel):
    """The semantic check's output shape.

    Previously a JsonOutputParser feeding a chain of .get() defaults, where an
    unrecognised severity silently became "medium" — downgrading a critical
    allergen finding to a warning.
    """

    recipe_checks: list[_RecipeCheckSchema] = Field(default_factory=list)


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
        # chat_model=True is required: with_structured_output is only
        # implemented on ChatOllama, not OllamaLLM. json_mode is off because the
        # bound schema supersedes format="json" — setting both fights over the
        # same Ollama parameter.
        self._llm = build_llm(
            provider=provider,
            model=model,
            temperature=0,
            json_mode=False,
            chat_model=True,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
        )
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

        # Step 3: LLM semantic check.
        # If it fails we must not claim a recipe is SAFE — the rule-based pass
        # only matches literal ingredient names, so the semantic layer is the
        # only thing catching "prosciutto is pork". Degrade to WARNING instead.
        has_something_to_catch = bool(
            avoid_foods or constraint_rules or restrictions
            or constraints.limit or intent.health_conditions
        )
        semantic_check_failed = False
        try:
            llm_issues_map = await asyncio.to_thread(
                self._llm_semantic_check, recipes, constraints, intent,
            )
        except SafetyCheckError:
            llm_issues_map = {}
            # Nothing to catch means nothing was missed — only degrade when
            # the user actually has restrictions or conditions in play.
            semantic_check_failed = has_something_to_catch

        # Step 4: Merge and determine verdicts
        recipe_verdicts: list[RecipeSafetyResult] = []
        safe_recipes: list[Recipe] = []

        for i, recipe in enumerate(recipes):
            all_issues = rule_issues_per_recipe[i] + llm_issues_map.get(recipe.name, [])

            if any(iss.severity in ("critical", "high") for iss in all_issues):
                verdict = SafetyVerdict.UNSAFE
            elif any(iss.severity == "medium" for iss in all_issues):
                verdict = SafetyVerdict.WARNING
            elif semantic_check_failed:
                # Rule-based checks passed, but the semantic layer never ran.
                # "Not proven unsafe" is not the same as "safe".
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
        if semantic_check_failed:
            summary_parts.append(
                "  INCOMPLETE: the hidden-ingredient check could not run, so these "
                "recipes were not fully verified against your restrictions."
            )
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
        system = SAFETY_CHECK_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "PARSED RECIPES:\n{recipes_json}\n\nPATIENT CONSTRAINTS:\n- Restrictions/allergies: {restrictions}\n- Foods to AVOID: {avoid_foods}\n- Foods to LIMIT: {limit_foods}\n- Health conditions: {health_conditions}\n- Meal instructions: {instructions}\n- Nutrition limits: {nutrition_constraints}"),
        ])
        return prompt | self._llm.with_structured_output(_SafetyCheckSchema)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _word_match(word: str, text: str) -> bool:
        """True if `word` occurs in `text` as a whole word.

        A multi-word avoid item ("raw shellfish products") also matches on its
        individual content words, so it still catches an ingredient listed as
        just "shellfish". Filler words are dropped so "raw"/"products" alone
        cannot flag everything. Over-flagging is the safe direction here: a
        false positive costs one rejected recipe, a false negative feeds
        someone an allergen.
        """
        if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
            return True

        parts = [w for w in re.split(r'\W+', word.lower()) if w not in _FILLER_WORDS and len(w) >= 4]
        # Guard on "nothing left to match", NOT on "fewer than two words left".
        # "raw shellfish products" reduces to the single token ["shellfish"],
        # so a <2 check made this return False and the whole multi-word fix a
        # no-op — the exact case it was written for.
        if not parts:
            return False
        return any(
            re.search(r'\b' + re.escape(p) + r'\b', text, re.IGNORECASE) for p in parts
        )

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
                # Previously `if len(avoid.split()) > 2: continue`, which
                # silently gave "avoid raw shellfish products" zero rule-based
                # checking and left only the LLM pass to catch it.
                if self._word_match(avoid, ing_lower):
                    issues.append(SafetyIssue(
                        category="avoid_food",
                        severity="critical",
                        description=f"Ingredient '{ingredient}' matches avoid item '{avoid}'",
                        detail=f"avoid_match: '{avoid}' in '{ingredient}'",
                    ))

            for restriction in restrictions_lower:
                # Skip if the ingredient explicitly declares itself as compliant
                # e.g. "gluten-free flour" should not violate "gluten-free"
                if restriction in ing_lower:
                    continue
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

    # DEF _CHECK_NUTRITION: 
    # | Task   | What is divided?            | Meaning                            |
    # | ------ | --------------------------- | ---------------------------------- |
    # | TASK 1 | Nutrition limit (`max_val`) | Convert daily → per meal           |
    # | TASK 2 | (removed — values already arrive per serving)             |
    # | TASK 3 | Interpretation change       | Treat constraints as daily         |

    def _check_nutrition(
        self,
        nutrition: NutritionValues,
        constraint_rules: dict[str, dict[str, float | None]],
    ) -> list[SafetyIssue]:

        issues: list[SafetyIssue] = []

        if nutrition is None:
            return issues

        if isinstance(nutrition, dict):
            nutrition = NutritionValues(**{k: v for k, v in nutrition.items() if k in NutritionValues.__dataclass_fields__})

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

            # NOTE: deliberately NOT divided by servings. The values are
            # already per-serving by contract — see _NutritionSchema in
            # recipe_rag.py ("kcal per serving") and the retrieval prompts.
            # Dividing again let a 1500 mg/serving sodium recipe pass a
            # 2000 mg/day limit as 1500/4 = 375, while the UI showed the
            # undivided 1500 under a "per serving" heading.
            max_val = rule.get("max")
            min_val = rule.get("min")

            # Assume daily constraints → convert to per-meal (1/3 of daily)
            if max_val is not None:     # If there's a max constraint, apply it as a per-meal limit (1/3 of daily)
                # TASK 1: Divide each nutrition value by 3
                allowed_max = max_val / 3       # Future: could be smarter about meal distribution (e.g., more calories at dinner), but 1/3 is a simple heuristic
                if value > allowed_max:         # If the recipe exceeds the per-meal limit, flag it (medium severity since it's not an outright violation, but a concern)
                    issues.append(SafetyIssue(          # Note: we use "medium" severity here to indicate it's a concern but not necessarily a critical violation, since it's based on a heuristic per-meal limit rather than an absolute max for the recipe.
                        category="nutrition_limit",         # In the future, we could have more nuanced categories like "nutrition_excess" vs "nutrition_deficiency"
                        severity="medium",                  # This is a judgment call — we want to flag it as a concern, but not necessarily reject the recipe outright, since it's based on a heuristic per-meal limit rather than an absolute max for the recipe. In a future iteration, we could consider making this "high" severity if it exceeds the daily max (without dividing by 3), and "medium" if it exceeds the per-meal heuristic.
                        description=f"{nutrient} per serving ({value:.2f}) exceeds per-meal limit ({allowed_max:.2f})",         # The description is human-readable and explains the concern clearly.
                        detail=f"{nutrient}: {value:.2f} > allowed {allowed_max:.2f}",      # The detail provides the specific values for debugging and transparency.
                    ))

            if min_val is not None:         # If there's a min constraint, we can also check if the recipe is too low in that nutrient (e.g., protein for a high-protein requirement). We can apply a similar per-meal heuristic (1/3 of daily minimum) for this as well.
                # TASK 1: Divide each nutrition value by 3
                required_min = min_val / 3          
                if value < required_min:            
                    issues.append(SafetyIssue(          
                        category="nutrition_limit",             
                        severity="medium",                      
                        description=f"{nutrient} per serving ({value:.2f}) below per-meal minimum ({required_min:.2f})",
                        detail=f"{nutrient}: {value:.2f} < required {required_min:.2f}",
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

            parsed = (
                result if isinstance(result, _SafetyCheckSchema)
                else _SafetyCheckSchema(**result)
            )

            issues_map: dict[str, list[SafetyIssue]] = {}
            for check in parsed.recipe_checks:
                issues = [
                    SafetyIssue(
                        category=iss.category,
                        severity=_normalize_severity(iss.severity),
                        description=iss.description,
                        detail=iss.detail,
                    )
                    for iss in check.issues
                ]
                if issues:
                    issues_map[check.recipe_name] = issues
            return issues_map
        except Exception as e:
            # Never swallow this. An empty map is indistinguishable from
            # "no problems found", which would silently pass unsafe recipes.
            logger.error("LLM semantic safety check failed: %s", e)
            raise SafetyCheckError("LLM semantic safety check failed") from e

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
            else:
                lines.append("\n**Nutrition:** not available for this recipe")
            parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts)


# Backward-compatible alias (so old imports still work)
OllamaSafetyFilter = SafetyFilter
