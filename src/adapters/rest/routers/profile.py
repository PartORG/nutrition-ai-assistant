"""Protected profile endpoint."""

from fastapi import APIRouter, Depends

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser, build_session_ctx
from adapters.rest.schemas import ProfileOut, MedicalAdviceOut, UserOut

router = APIRouter(tags=["profile"])


@router.get("/profile")
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


async def _gather(*coros):
    """Run multiple coroutines concurrently."""
    import asyncio
    return await asyncio.gather(*coros)
