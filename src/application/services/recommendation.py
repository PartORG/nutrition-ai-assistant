"""
application.services.recommendation - Core recommendation pipeline.

Orchestrates the 5-step AI flow:
    1. Parse user intent (LLM)
    2. Get medical constraints (Medical RAG)
    3. Build augmented query
    4. Retrieve recipe recommendations (Recipe RAG)
    5. Safety check (mandatory, not optional)

Migrated from pipeline/pipeline.py with these removals:
    - No _step_handle_user (adapters manage users)
    - No _step_update_user (separate ProfileService)
    - No UserDBHandler construction or create_all_tables()
    - No print() debugging (uses logging)

All methods are async. Dependencies are injected via constructor.
"""

from __future__ import annotations

import logging
from typing import Optional

from domain.models import (
    UserIntent,
    NutritionConstraints,
    Recipe,
)
from domain.ports import (
    IntentParserPort,
    MedicalRAGPort,
    RecipeRAGPort,
    SafetyFilterPort,
    MedicalRepository,
    NutritionRepository,
)
from domain.exceptions import RAGError
from application.context import SessionContext
from application.dto import RecommendationResult

logger = logging.getLogger(__name__)


class RecommendationService:
    """Orchestrates the recipe recommendation pipeline.

    Pure orchestration — no persistence writes, no user management.
    All dependencies are injected. Stateless per call.
    """

    def __init__(
        self,
        intent_parser: IntentParserPort,
        medical_rag: MedicalRAGPort,
        recipe_rag: RecipeRAGPort,
        safety_filter: SafetyFilterPort,
        medical_repo: MedicalRepository,
        nutrition_repo: NutritionRepository,
    ):
        self._intent_parser = intent_parser
        self._medical_rag = medical_rag
        self._recipe_rag = recipe_rag
        self._safety_filter = safety_filter
        self._medical_repo = medical_repo
        self._nutrition_repo = nutrition_repo

    async def get_recommendations(
        self,
        ctx: SessionContext,
        query: str,
    ) -> RecommendationResult:
        """Run the full 5-step pipeline for a user query.

        Args:
            ctx:   Session context with user_id and user_data.
            query: The user's natural language request.

        Returns:
            RecommendationResult with all intermediate and final outputs.
        """
        logger.info(
            "Processing query for user %d (request=%s): %s",
            ctx.user_id, ctx.request_id, query[:80],
        )

        # Step 1: Parse intent
        intent = await self._parse_intent(query)
        logger.debug("Intent parsed: %s", intent)

        # Step 2: Get medical constraints (DB-first, then Medical RAG)
        constraints = await self._get_constraints(ctx, intent)
        logger.debug(
            "Constraints: avoid=%s, limits=%s",
            constraints.avoid,
            list(constraints.constraints.keys()),
        )

        # Step 2.5: Adjust constraints for today's already-consumed nutrition.
        # The limits in medical_advice are DAILY totals (e.g. sugar_g=50g/day).
        # We subtract what the user has already eaten today so that recipe
        # recommendations and safety checks are based on the REMAINING budget,
        # not the full daily limit.
        daily_totals = await self._get_daily_totals(ctx.user_id)
        adjusted_constraints = _adjust_for_daily_budget(constraints, daily_totals)
        if daily_totals:
            logger.info(
                "Daily totals for user %d: %s",
                ctx.user_id,
                {k: round(v, 1) for k, v in daily_totals.items() if v > 0},
            )

        # Step 3: Build augmented query (includes remaining daily budget)
        augmented = self._build_augmented_query(
            query, intent, adjusted_constraints, daily_totals,
        )
        logger.debug("Augmented query built (%d chars)", len(augmented))

        # Step 4: Retrieve recipes
        raw_recommendations = await self._retrieve_recipes(augmented)
        logger.debug("Recommendations received")

        # Step 5: Safety check against REMAINING daily budget
        safety_result = await self._safety_check(
            raw_recommendations, adjusted_constraints, intent,
        )
        logger.info(
            "Pipeline complete: %d/%d recipes passed safety",
            safety_result.safe_count, safety_result.total_count,
        )
        if safety_result.total_count > 0 and safety_result.safe_count == 0:
            logger.warning("Safety filter rejected ALL recipes. Details:\n%s", safety_result.summary)

        return RecommendationResult(
            intent=intent,
            constraints=adjusted_constraints,
            augmented_query=augmented,
            raw_recommendations=raw_recommendations,
            safety_result=safety_result,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _parse_intent(self, query: str) -> UserIntent:
        """Step 1: Extract structured intent from user query."""
        return await self._intent_parser.parse(query)

    async def _get_constraints(
        self,
        ctx: SessionContext,
        intent: UserIntent,
    ) -> NutritionConstraints:
        """Step 2: Get medical constraints.

        Merges health conditions from the current intent with those saved in
        the user's profile (loaded at session start into ctx.user_data).
        Checks DB cache for returning users, falls back to Medical RAG and
        saves the result for future caching.
        Always applies saved dietary restrictions/avoid from the user's profile.
        """
        # Merge conditions from current intent + saved profile (preserve order, dedup)
        saved_conditions = ctx.user_data.get("health_conditions", [])
        all_conditions = list(dict.fromkeys(intent.health_conditions + saved_conditions))

        if not all_conditions:
            # No health conditions — still apply saved profile restrictions/avoid
            return self._apply_profile_data(NutritionConstraints.default(), ctx)

        # Check if we have cached constraints for this user
        cached_advice = await self._medical_repo.get_by_user(ctx.user_id)
        if cached_advice:
            latest = cached_advice[0]  # newest first
            if latest.medical_advice:
                logger.debug("Using cached medical advice for user %d", ctx.user_id)
                constraints = NutritionConstraints(
                    notes=latest.medical_advice,
                    avoid=_split_or_empty(latest.avoid),
                    limit=_split_or_empty(latest.dietary_limit),
                    constraints=_parse_constraints_str(latest.dietary_constraints),
                )
                return self._apply_profile_data(constraints, ctx)

        # Fall back to Medical RAG — gracefully degrade if it's unavailable
        try:
            constraints = await self._medical_rag.get_constraints(all_conditions)
        except Exception:
            logger.exception(
                "Medical RAG failed for user %d (conditions=%s) — using default constraints",
                ctx.user_id, all_conditions,
            )
            return self._apply_profile_data(NutritionConstraints.default(), ctx)

        # Persist RAG result for future caching (non-blocking on failure)
        await self._save_medical_advice_to_db(ctx, all_conditions, constraints)

        return self._apply_profile_data(constraints, ctx)

    def _apply_profile_data(
        self, constraints: NutritionConstraints, ctx: SessionContext,
    ) -> NutritionConstraints:
        """Merge saved profile restrictions/avoid foods into constraints.

        Ensures that foods the user must avoid (from their medical history) and
        dietary restrictions (e.g. vegan, gluten-free from their profile) are
        always included, even if they weren't mentioned in the current message.
        """
        saved_restrictions = ctx.user_data.get("restrictions", [])
        raw_avoid = ctx.user_data.get("avoid", [])

        # user_data["avoid"] may be list[str] where each item is comma-separated
        flat_avoid: list[str] = []
        for item in raw_avoid:
            if isinstance(item, str):
                flat_avoid.extend(_split_or_empty(item))
            else:
                flat_avoid.append(str(item))

        merged_avoid = list(dict.fromkeys(constraints.avoid + flat_avoid))
        merged_limit = list(dict.fromkeys(constraints.limit + saved_restrictions))

        if merged_avoid == constraints.avoid and merged_limit == constraints.limit:
            return constraints

        return NutritionConstraints(
            dietary_goals=constraints.dietary_goals,
            foods_to_increase=constraints.foods_to_increase,
            avoid=merged_avoid,
            limit=merged_limit,
            constraints=constraints.constraints,
            notes=constraints.notes,
        )

    async def _save_medical_advice_to_db(
        self,
        ctx: SessionContext,
        conditions: list[str],
        constraints: NutritionConstraints,
    ) -> None:
        """Persist Medical RAG result to DB so it can be cached on next request."""
        import json
        from domain.entities import MedicalAdvice

        advice = MedicalAdvice(
            user_id=ctx.user_id,
            health_condition=", ".join(conditions),
            medical_advice=constraints.notes,
            avoid=", ".join(constraints.avoid),
            dietary_limit=", ".join(constraints.limit),
            dietary_constraints=json.dumps(constraints.constraints) if constraints.constraints else "",
        )
        try:
            await self._medical_repo.save(advice)
            logger.info("Saved medical advice to DB for user %d", ctx.user_id)
        except Exception:
            logger.exception("Failed to save medical advice — continuing without save")

    async def _get_daily_totals(self, user_id: int) -> dict[str, float]:
        """Step 2.5: Sum today's already-consumed nutrition for the user.

        Returns a dict mapping constraint key names to total amounts consumed
        so far today.  Returns an empty dict if no meals were saved today or
        if the query fails.
        """
        try:
            records = await self._nutrition_repo.get_today_by_user(user_id)
        except Exception:
            logger.exception("Failed to load today's nutrition — skipping budget adjustment")
            return {}

        if not records:
            return {}

        totals: dict[str, float] = {
            "calories": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "fiber_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
        }
        for r in records:
            totals["calories"] += r.calories or 0.0
            totals["protein_g"] += r.protein or 0.0
            totals["fat_g"] += r.fat or 0.0
            totals["carbs_g"] += r.carbohydrates or 0.0
            totals["fiber_g"] += r.fiber or 0.0
            totals["sugar_g"] += r.sugar or 0.0
            totals["sodium_mg"] += r.sodium or 0.0

        return totals

    async def _retrieve_recipes(self, augmented_query: str) -> list[Recipe]:
        """Step 4: Send augmented query to Recipe RAG."""
        result = await self._recipe_rag.async_ask(augmented_query)
        if not result:
            raise RAGError("Recipe RAG returned empty response")
        return result

    async def _safety_check(
        self,
        raw_recommendations: list[Recipe],
        constraints: NutritionConstraints,
        intent: UserIntent,
    ):
        """Step 5: Validate recipes against user constraints."""
        return await self._safety_filter.check(
            recipes=raw_recommendations,
            constraints=constraints,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Query augmentation (Step 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_augmented_query(
        original_query: str,
        intent: UserIntent,
        constraints: NutritionConstraints,
        daily_totals: Optional[dict] = None,
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
            rules = constraints.constraints  # already reduced by _adjust_for_daily_budget

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

def _adjust_for_daily_budget(
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


def _split_or_empty(value: Optional[str]) -> list[str]:
    """Split a comma/newline-separated string into a list, or return []."""
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _parse_constraints_str(value: Optional[str]) -> dict:
    """Parse a JSON-like constraints string, or return empty dict."""
    if not value:
        return {}
    import json
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
