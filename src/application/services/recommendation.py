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
    ):
        self._intent_parser = intent_parser
        self._medical_rag = medical_rag
        self._recipe_rag = recipe_rag
        self._safety_filter = safety_filter
        self._medical_repo = medical_repo

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

        # Step 2: Get medical constraints
        constraints = await self._get_constraints(ctx, intent)
        logger.debug(
            "Constraints: avoid=%s, limits=%s",
            constraints.avoid,
            list(constraints.constraints.keys()),
        )

        # Step 3: Build augmented query
        augmented = self._build_augmented_query(query, intent, constraints)
        logger.debug("Augmented query built (%d chars)", len(augmented))

        # Step 4: Retrieve recipes
        raw_recommendations = await self._retrieve_recipes(augmented)
        logger.debug("Recommendations received")

        # Step 5: Safety check (mandatory — not optional, not a tool call)
        safety_result = await self._safety_check(
            raw_recommendations, constraints, intent,
        )
        logger.info(
            "Pipeline complete: %d/%d recipes passed safety",
            safety_result.safe_count, safety_result.total_count,
        )

        return RecommendationResult(
            intent=intent,
            constraints=constraints,
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

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
