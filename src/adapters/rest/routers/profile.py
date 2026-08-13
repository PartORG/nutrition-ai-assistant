"""Protected profile endpoint."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from adapters.rest.dependencies import CurrentUser, build_session_ctx, get_current_user, get_factory
from adapters.rest.schemas import MedicalAdviceOut, ProfileOut, UserOut
from factory import ServiceFactory

router = APIRouter(tags=["profile"])


class UserUpdateRequest(BaseModel):
    """Request body for updating user profile."""
    name: str
    age: int
    gender: str
    caretaker: str


class HealthUpdateRequest(BaseModel):
    """Request body for updating health conditions."""
    health_condition: str


class DietaryConstraintsUpdateRequest(BaseModel):
    """Request body for updating dietary constraints."""
    dietary_constraints: str


class PreferencesUpdateRequest(BaseModel):
    """Request body for updating food preferences."""
    preferences: str


class RestrictionsUpdateRequest(BaseModel):
    """Request body for updating dietary restrictions / dislikes."""
    restrictions: str


async def _gather(*coros):
    """Run multiple coroutines concurrently."""
    return await asyncio.gather(*coros)


@router.get("")
async def get_profile(
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_profile_service()
    user_repo = factory.create_user_repository()
    ctx = await build_session_ctx(user.user_id, "", factory)

    user_entity, profiles, medical = await _gather(
        user_repo.get_by_id(user.user_id),
        service.get_profile_history(ctx),
        service.get_medical_advice(ctx),
    )

    return {
        "user": UserOut(
            name=user_entity.name if user_entity else "",
            surname=user_entity.surname if user_entity else "",
            user_name=user_entity.user_name if user_entity else "",
            age=user_entity.age if user_entity else 0,
            gender=user_entity.gender if user_entity else "",
            caretaker=user_entity.caretaker if user_entity else "",
        ),
        "profiles": [
            ProfileOut(
                preferences=p.preferences,
                health_condition=p.health_condition,
                restrictions=p.restrictions,
                created_at=p.created_at,
            )
            for p in profiles
        ],
        "medical_advice": [
            MedicalAdviceOut(
                health_condition=m.health_condition,
                medical_advice=m.medical_advice,
                dietary_limit=m.dietary_limit,
                avoid=m.avoid,
                dietary_constraints=m.dietary_constraints,
            )
            for m in medical
        ],
    }


@router.post("/update")
async def update_profile(
    data: UserUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update user profile information."""
    user_repo = factory.create_user_repository()

    # Checked BEFORE the writes: a 24h token outlives account deletion, and
    # updating a deleted user's row and then 404-ing is the worst of both.
    if await user_repo.get_by_id(user.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Update each field individually
    await user_repo.update(user.user_id, "name", data.name)
    await user_repo.update(user.user_id, "age", data.age)
    await user_repo.update(user.user_id, "gender", data.gender)
    await user_repo.update(user.user_id, "caretaker", data.caretaker)

    user_entity = await user_repo.get_by_id(user.user_id)
    if user_entity is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "message": "Profile updated successfully",
        "user": UserOut(
            name=user_entity.name,
            surname=user_entity.surname,
            user_name=user_entity.user_name,
            age=user_entity.age,
            gender=user_entity.gender,
            caretaker=user_entity.caretaker,
        )
    }


@router.post("/update-health")
async def update_health(
    data: HealthUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update health conditions in the latest profile history."""
    from domain.entities import UserProfileHistory
    from infrastructure.persistence.medical_repo import SQLiteMedicalRepository
    from infrastructure.persistence.profile_repo import SQLiteProfileRepository
    profile_repo = SQLiteProfileRepository(factory._connection)

    profiles = await profile_repo.get_by_user(user.user_id)
    if not profiles:
        await profile_repo.save(UserProfileHistory(
            user_id=user.user_id, health_condition=data.health_condition,
        ))
    else:
        await profile_repo.update_field(profiles[0].id, "health_condition", data.health_condition)

    # Invalidate medical advice cache so RAG re-runs on the next recommendation
    # request with the updated health conditions.
    # We clear the generated text fields (medical_advice, avoid, dietary_limit)
    # so the cache-miss path triggers a fresh RAG run, while preserving any
    # dietary_constraints the user has manually edited (the CASE in
    # update_advice_fields keeps non-empty dietary_constraints intact).
    medical_repo = SQLiteMedicalRepository(factory._connection)
    existing_advice = await medical_repo.get_by_user(user.user_id)
    if existing_advice:
        await medical_repo.update_advice_fields(
            existing_advice[0].id,
            health_condition=data.health_condition,
            medical_advice="",
            avoid="",
            dietary_limit="",
            dietary_constraints="",  # preserved by SQL CASE if already set
        )

    return {"message": "Health conditions updated successfully"}


@router.post("/update-preferences")
async def update_preferences(
    data: PreferencesUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update food preferences in the latest profile history."""
    from domain.entities import UserProfileHistory
    from infrastructure.persistence.profile_repo import SQLiteProfileRepository
    profile_repo = SQLiteProfileRepository(factory._connection)

    profiles = await profile_repo.get_by_user(user.user_id)
    if not profiles:
        await profile_repo.save(UserProfileHistory(
            user_id=user.user_id, preferences=data.preferences,
        ))
    else:
        await profile_repo.update_field(profiles[0].id, "preferences", data.preferences)

    return {"message": "Preferences updated successfully"}


@router.post("/update-restrictions")
async def update_restrictions(
    data: RestrictionsUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update dietary restrictions / dislikes in the latest profile history."""
    from domain.entities import UserProfileHistory
    from infrastructure.persistence.profile_repo import SQLiteProfileRepository
    profile_repo = SQLiteProfileRepository(factory._connection)

    profiles = await profile_repo.get_by_user(user.user_id)
    if not profiles:
        await profile_repo.save(UserProfileHistory(
            user_id=user.user_id, restrictions=data.restrictions,
        ))
    else:
        await profile_repo.update_field(profiles[0].id, "restrictions", data.restrictions)

    return {"message": "Restrictions updated successfully"}


@router.post("/update-dietary-constraints")
async def update_dietary_constraints(
    data: DietaryConstraintsUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update dietary constraints in the latest medical advice."""
    from domain.entities import MedicalAdvice
    from infrastructure.persistence.medical_repo import SQLiteMedicalRepository
    medical_repo = SQLiteMedicalRepository(factory._connection)

    advices = await medical_repo.get_by_user(user.user_id)
    if not advices:
        # Create a new medical advice entry
        new_advice = MedicalAdvice(
            health_condition="",
            medical_advice="",
            dietary_limit="",
            avoid="",
            dietary_constraints=data.dietary_constraints,
            user_id=user.user_id,
        )
        await medical_repo.save(new_advice)
    else:
        latest = advices[0]
        await medical_repo.update_field(latest.id, "dietary_constraints", data.dietary_constraints)

    return {"message": "Dietary constraints updated successfully"}