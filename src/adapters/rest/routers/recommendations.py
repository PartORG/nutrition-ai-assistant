"""Protected recommendation endpoint."""

from fastapi import APIRouter, Depends

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser, build_session_ctx
from adapters.rest.schemas import RecommendationBody, RecommendationOut

router = APIRouter(tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationOut)
async def get_recommendations(
    body: RecommendationBody,
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_recommendation_service()
    ctx = await build_session_ctx(user.user_id, "rest-recommendation", factory)
    result = await service.get_recommendations(ctx, body.query)
    return RecommendationOut(
        summary=result.summary or "",
        raw_recommendations=result.safety_result.safe_recipes_markdown,
    )
