"""Protected image analysis endpoint (file upload)."""

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from adapters.rest.dependencies import CurrentUser, build_session_ctx, get_current_user, get_factory
from adapters.rest.schemas import ImageAnalysisOut
from factory import ServiceFactory

router = APIRouter(tags=["images"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# ~40 MP is well past any phone camera and bounds the decoded RGB buffer to
# roughly 120 MB, which both detectors can survive concurrently.
MAX_UPLOAD_PIXELS = 40_000_000

# Only what the detectors can actually decode. The stored *basename* is always
# a uuid4 — only the extension comes from the client, and only from this set.
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _get_upload_dir() -> Path:
    """Return the configured upload directory, creating it if necessary.

    Uses UPLOAD_DIR env var so Docker and local runs both work:
      - Locally:  ./uploads/ (relative to cwd, i.e. the project root)
      - Docker:   /app/uploads
    """
    upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


async def _save_upload(file: UploadFile) -> Path:
    """Persist an uploaded image under the upload dir, or raise 4xx.

    Bounds what lands in uploads/, writes off the event loop so a large write
    cannot stall other requests, and verifies the bytes really are a decodable
    image of sane dimensions. (The whole request body has already been spooled
    to disk by Starlette before this runs — the body cap lives in app.py.)
    """
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported image type '{suffix}'.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")

    dest = _get_upload_dir() / f"{uuid.uuid4().hex}{suffix}"
    await asyncio.to_thread(dest.write_bytes, data)

    try:
        await asyncio.to_thread(_verify_image, dest)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def _verify_image(path: Path) -> None:
    """Reject non-images and decompression bombs.

    The suffix allowlist says nothing about the bytes, and neither YOLO's
    cv2.imread nor the VLM guards against a bomb: a 13000x13000 uniform PNG
    compresses to well under the 10 MB cap and decodes to ~484 MB of RGB.
    Image.open reads the header only, so this stays cheap.
    """
    try:
        with Image.open(path) as im:
            pixels = im.width * im.height
            fmt = im.format
    except Exception as exc:
        raise HTTPException(400, "File is not a decodable image.") from exc

    if fmt not in {"JPEG", "PNG", "WEBP"}:
        raise HTTPException(400, f"Unsupported image format '{fmt}'.")
    if pixels > MAX_UPLOAD_PIXELS:
        raise HTTPException(413, "Image resolution too large (max 40 MP).")


@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Upload an image and get back a server-side path for use in WebSocket chat.

    The returned path is valid on the server. Pass it to the agent via a
    WebSocket message so the analyze_image tool can open it.
    """
    dest = await _save_upload(file)
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

    # Save upload to the upload directory so the detector can read the file.
    tmp_path = str((await _save_upload(file)).resolve())

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
