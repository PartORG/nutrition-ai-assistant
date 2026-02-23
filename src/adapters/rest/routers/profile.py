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
    
    # Update user entity
    user_entity = await user_repo.get_by_id(user.user_id)
    if not user_entity:
        print(f"🔵 Backend: User not found")
        return {"error": "User not found"}, 404
    
    user_entity.name = data.name
    user_entity.age = data.age
    user_entity.gender = data.gender
    user_entity.caretaker = data.caretaker
    
    await user_repo.update(user_entity)
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