"""Protected analytics endpoint — aggregated usage statistics."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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


class _RatingUpdate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")


@router.patch("/dashboard/recipes/{recipe_id}/rating")
async def update_recipe_rating(
    recipe_id: int,
    body: _RatingUpdate,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    """Update the star rating for a saved recipe (1–5)."""
    repo = factory.create_analytics_repository()
    updated = await repo.update_recipe_rating(user.user_id, recipe_id, body.rating)
    if not updated:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {"ok": True}


async def _gather(repo):
    from asyncio import gather
    return await gather(
        repo.get_overview(),
        repo.get_top_recipes(limit=5),
        repo.get_common_conditions(limit=5),
        repo.get_recent_conversations(limit=10),
    )
