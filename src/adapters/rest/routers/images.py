"""Protected image analysis endpoint (file upload)."""

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from factory import ServiceFactory
from adapters.rest.dependencies import get_factory, get_current_user, CurrentUser, build_session_ctx
from adapters.rest.schemas import ImageAnalysisOut

router = APIRouter(tags=["images"])

# Persistent upload directory — files stay until the agent processes them
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "nutriai_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Upload an image and get back a server-side path for use in WebSocket chat.

    The returned path is valid on the server and can be passed to the agent
    via a WebSocket message so the analyze_image tool can open it.
    """
    suffix = Path(file.filename or "image.jpg").suffix
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = _UPLOAD_DIR / filename
    dest.write_bytes(await file.read())
    return JSONResponse({"path": str(dest)})


@router.post("/image/analyze", response_model=ImageAnalysisOut)
async def analyze_image(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_image_analysis_service()
    ctx = await build_session_ctx(user.user_id, "rest-image", factory)

    # Save upload to temp file for CNN detector
    suffix = Path(file.filename or "image.jpg").suffix
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await service.recommend_from_image(ctx, tmp_path)

        return ImageAnalysisOut(
            detected_ingredients=[i for i in result.detected.ingredients],
            recommendation_summary=(
                result.recommendation.summary
                if result.recommendation else None
            ),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
