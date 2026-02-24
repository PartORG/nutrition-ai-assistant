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


def _get_upload_dir() -> Path:
    """Return the configured upload directory, creating it if necessary.

    Uses UPLOAD_DIR env var so Docker and local runs both work:
      - Locally:  ./uploads/ (relative to cwd, i.e. the project root)
      - Docker:   /app/uploads  (shared volume between api and yolo-detector)
    """
    upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Upload an image and get back a server-side path for use in WebSocket chat.

    The returned path is valid on the server AND on the YOLO service (both
    containers share the same ./uploads volume mount).  Pass it to the agent
    via a WebSocket message so the analyze_image tool can open it.
    """
    upload_dir = _get_upload_dir()
    suffix = Path(file.filename or "image.jpg").suffix
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = upload_dir / filename
    dest.write_bytes(await file.read())
    # Return the absolute path so it works regardless of cwd
    return JSONResponse({"path": str(dest.resolve())})


@router.post("/image/analyze", response_model=ImageAnalysisOut)
async def analyze_image(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    factory: ServiceFactory = Depends(get_factory),
):
    service = factory.create_image_analysis_service()
    ctx = await build_session_ctx(user.user_id, "rest-image", factory)

    # Save upload to the shared upload directory so the YOLO service can also
    # read the file via the shared volume mount.
    upload_dir = _get_upload_dir()
    suffix = Path(file.filename or "image.jpg").suffix
    dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(await file.read())
    tmp_path = str(dest.resolve())

    try:
        result = await service.recommend_from_image(ctx, tmp_path)

        return ImageAnalysisOut(
            detected_ingredients=[i for i in result.detected.ingredients],
            recommendation_summary=(
                result.recommendation.summary
                if result.recommendation else None
            ),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
