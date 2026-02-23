"""Protected profile endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser, build_session_ctx
from adapters.rest.schemas import ProfileOut, MedicalAdviceOut, UserOut

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
    print(f"🔵 Backend: Received update request for user {user.user_id}")
    print(f"🔵 Backend: Data = {data}")
    
    user_repo = factory.create_user_repository()
    
    # Update each field individually
    await user_repo.update(user.user_id, "name", data.name)
    await user_repo.update(user.user_id, "age", data.age)
    await user_repo.update(user.user_id, "gender", data.gender)
    await user_repo.update(user.user_id, "caretaker", data.caretaker)
    
    # Fetch updated user
    user_entity = await user_repo.get_by_id(user.user_id)
    print(f"🔵 Backend: User updated successfully")
    
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
    from infrastructure.persistence.profile_repo import SQLiteProfileRepository
    profile_repo = SQLiteProfileRepository(factory._connection)

    profiles = await profile_repo.get_by_user(user.user_id)
    if not profiles:
        return {"error": "No profile found"}

    latest = profiles[0]
    await profile_repo.update_field(latest.id, "health_condition", data.health_condition)

    return {"message": "Health conditions updated successfully"}


@router.post("/update-dietary-constraints")
async def update_dietary_constraints(
    data: DietaryConstraintsUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update dietary constraints in the latest medical advice."""
    from infrastructure.persistence.medical_repo import SQLiteMedicalRepository
    from domain.entities import MedicalAdvice
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