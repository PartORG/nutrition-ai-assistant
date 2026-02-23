"""Protected analytics endpoint — aggregated usage statistics."""

from fastapi import APIRouter, Depends

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser

router = APIRouter(tags=["analytics"])


@router.get("/analytics")
async def get_analytics(
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """
    Return aggregated platform statistics.

    Available to any authenticated user. Shows:
    - Overview counts (users, conversations, messages, saved recipes)
    - Top 5 most saved recipes
    - Top 5 most common health conditions
    - 10 most recently active conversations
    """
    repo = factory.create_analytics_repository()
    overview, top_recipes, conditions, recent = await _gather(repo)
    return {
        "overview": overview,
        "top_recipes": top_recipes,
        "common_conditions": conditions,
        "recent_conversations": recent,
    }


@router.get("/dashboard")
async def get_dashboard(
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """
    Return per-user dashboard data.

    Shows the current user's own:
    - Overview counts (conversations, messages, saved recipes)
    - Average nutrition values across all saved meals
    - 10 most recently saved recipes with nutrition
    """
    repo = factory.create_analytics_repository()
    return await repo.get_user_dashboard(user.user_id)


async def _gather(repo):
    from asyncio import gather
    return await gather(
        repo.get_overview(),
        repo.get_top_recipes(limit=5),
        repo.get_common_conditions(limit=5),
        repo.get_recent_conversations(limit=10),
    )
