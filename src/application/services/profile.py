"""
application.services.profile - User profile and medical advice management.

Extracted from pipeline.py (_step_update_user) and medical advice persistence.
Handles profile snapshots and medical advice caching — separate from the
recommendation pipeline.
"""

from __future__ import annotations

import logging

from domain.models import UserIntent
from domain.entities import UserProfileHistory, MedicalAdvice
from domain.ports import ProfileRepository, MedicalRepository
from application.context import SessionContext

logger = logging.getLogger(__name__)


class ProfileService:
    """Manages user dietary profile snapshots and medical advice."""

    def __init__(
        self,
        profile_repo: ProfileRepository,
        medical_repo: MedicalRepository,
    ):
        self._profile_repo = profile_repo
        self._medical_repo = medical_repo

    async def save_initial_profile(
        self,
        user_id: int,
        health_condition: str,
    ) -> None:
        """Persist health conditions provided at registration time.

        Called by the auth router immediately after a new user is created so
        that load_user_context() can pick them up on first login.
        Skips silently if health_condition is empty.
        """
        conditions = [c.strip() for c in health_condition.split(",") if c.strip()]
        if not conditions:
            return
        profile = UserProfileHistory(
            user_id=user_id,
            health_condition=", ".join(conditions),
            preferences="",
            restrictions="",
        )
        await self._profile_repo.save(profile)
        logger.debug(
            "Saved initial health profile for user %d: %s",
            user_id, conditions,
        )

    async def update_profile(
        self,
        ctx: SessionContext,
        intent: UserIntent,
    ) -> None:
        """Save the user's current profile as a new snapshot.

        Called after intent parsing to track profile changes over time.
        """
        logger.debug(
            "Saving profile snapshot for user %d: conditions=%s, restrictions=%s",
            ctx.user_id, intent.health_conditions, intent.restrictions,
        )

        profile = UserProfileHistory(
            user_id=ctx.user_id,
            preferences=", ".join(intent.preferences),
            health_condition=", ".join(intent.health_conditions),
            restrictions=", ".join(intent.restrictions),
        )
        await self._profile_repo.save(profile)

    async def save_medical_advice(
        self,
        ctx: SessionContext,
        health_condition: str,
        advice_text: str,
        avoid: str = "",
        dietary_limit: str = "",
        dietary_constraints: str = "",
    ) -> int:
        """Persist medical advice from RAG for future caching."""
        advice = MedicalAdvice(
            user_id=ctx.user_id,
            health_condition=health_condition,
            medical_advice=advice_text,
            avoid=avoid,
            dietary_limit=dietary_limit,
            dietary_constraints=dietary_constraints,
        )
        return await self._medical_repo.save(advice)

    async def get_medical_advice(
        self,
        ctx: SessionContext,
    ) -> list[MedicalAdvice]:
        """Retrieve all medical advice records for the user (newest first)."""
        return await self._medical_repo.get_by_user(ctx.user_id)

    async def get_profile_history(
        self,
        ctx: SessionContext,
    ) -> list[UserProfileHistory]:
        """Retrieve all profile snapshots for the user (newest first)."""
        return await self._profile_repo.get_by_user(ctx.user_id)

    async def load_user_context(self, user_id: int) -> dict:
        """Build a user_data dict from the latest profile + medical advice.

        Used by REST/WS adapters to pre-populate SessionContext so the
        recommendation pipeline already knows the user's conditions and
        preferences without the user having to re-state them every request.

        Returns a dict with keys: preferences, health_conditions, restrictions, avoid.
        Empty dict if the user has no profile history yet.
        """
        profiles = await self._profile_repo.get_by_user(user_id)
        medical = await self._medical_repo.get_by_user(user_id)

        user_data: dict = {}

        if profiles:
            latest = profiles[0]  # ordered DESC — most recent first
            user_data["preferences"] = [
                p.strip() for p in latest.preferences.split(",") if p.strip()
            ]
            user_data["health_conditions"] = [
                h.strip() for h in latest.health_condition.split(",") if h.strip()
            ]
            user_data["restrictions"] = [
                r.strip() for r in latest.restrictions.split(",") if r.strip()
            ]

        if medical:
            avoid = [m.avoid for m in medical if m.avoid]
            if avoid:
                user_data["avoid"] = avoid

        logger.debug(
            "Loaded user context for user %d: %s",
            user_id, list(user_data.keys()),
        )
        return user_data
