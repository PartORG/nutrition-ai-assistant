"""
The image path arrives from the user and reaches the filesystem without an
LLM in the loop: `[IMAGE:/etc/passwd]` in a chat message is regex-extracted
by executor._route_image_if_present and handed straight to a detector.

These tests pin the containment guard AND the cleanup that follows it. An
earlier version of the cleanup recorded the path before validating it, which
turned a blocked read into an arbitrary file delete.
"""

from __future__ import annotations

import pytest

from agent.tools.analyze_image import AnalyzeImageTool
from application.context import SessionContext
from application.services.image_analysis import ImageAnalysisService
from domain.exceptions import IngredientDetectionError
from domain.models import DetectedIngredients


class _Detector:
    def __init__(self):
        self.called_with = []

    async def detect(self, image_path: str) -> DetectedIngredients:
        self.called_with.append(image_path)
        return DetectedIngredients(ingredients=["apple"], image_path=image_path)


class _FailingDetector:
    async def detect(self, image_path: str) -> DetectedIngredients:
        raise IngredientDetectionError("ollama unreachable")


def _ctx() -> SessionContext:
    return SessionContext(user_id=1, conversation_id="c")


@pytest.fixture
def uploads(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


async def test_path_outside_upload_dir_is_rejected(tmp_path, uploads):
    secret = tmp_path / "users.db"
    secret.write_text("secret")
    det = _Detector()
    svc = ImageAnalysisService(det, None, upload_dir=str(uploads))

    with pytest.raises(IngredientDetectionError, match="not an uploaded file"):
        await svc.detect_ingredients(_ctx(), str(secret))

    assert det.called_with == []      # never reached the detector
    assert secret.exists()            # and was not deleted


async def test_traversal_out_of_upload_dir_is_rejected(tmp_path, uploads):
    secret = tmp_path / "users.db"
    secret.write_text("secret")
    svc = ImageAnalysisService(_Detector(), None, upload_dir=str(uploads))

    with pytest.raises(IngredientDetectionError):
        await svc.detect_ingredients(_ctx(), f"{uploads}/../users.db")


async def test_symlink_escaping_the_upload_dir_is_rejected(tmp_path, uploads):
    secret = tmp_path / "users.db"
    secret.write_text("secret")
    link = uploads / "innocent.jpg"
    link.symlink_to(secret)
    svc = ImageAnalysisService(_Detector(), None, upload_dir=str(uploads))

    with pytest.raises(IngredientDetectionError):
        await svc.detect_ingredients(_ctx(), str(link))


async def test_sibling_directory_sharing_a_prefix_is_rejected(tmp_path, uploads):
    """`uploads_evil/` must not pass a naive startswith-style check."""
    evil = tmp_path / "uploads_evil"
    evil.mkdir()
    target = evil / "x.jpg"
    target.write_bytes(b"x")
    svc = ImageAnalysisService(_Detector(), None, upload_dir=str(uploads))

    with pytest.raises(IngredientDetectionError):
        await svc.detect_ingredients(_ctx(), str(target))


async def test_legitimate_upload_is_accepted(uploads):
    img = uploads / "abc123.jpg"
    img.write_bytes(b"x")
    det = _Detector()
    svc = ImageAnalysisService(det, None, upload_dir=str(uploads))

    result = await svc.detect_ingredients(_ctx(), str(img))

    assert result.ingredients == ["apple"]
    assert det.called_with == [str(img)]


# ── the cleanup must never delete what the guard refused ───────────────────

async def test_rejected_path_is_not_deleted_by_the_cleanup(tmp_path, uploads):
    """Regression: the finally-block used to unlink the attacker's path."""
    victim = tmp_path / "users.db"
    victim.write_text("PRECIOUS")
    svc = ImageAnalysisService(_Detector(), None, upload_dir=str(uploads))
    tool = AnalyzeImageTool(svc)
    ctx = _ctx()
    ctx.scratch["original_query"] = f"what is this [IMAGE:{victim}]"

    with pytest.raises(IngredientDetectionError):
        await tool.execute(ctx, image_path=str(victim))

    assert victim.exists(), "guard blocked the read but cleanup deleted the file"


async def test_upload_survives_a_transient_detector_failure(uploads):
    """A failed detection must leave the photo so the retry can work."""
    img = uploads / "abc123.jpg"
    img.write_bytes(b"x")
    svc = ImageAnalysisService(_FailingDetector(), None, upload_dir=str(uploads))
    tool = AnalyzeImageTool(svc)
    ctx = _ctx()
    ctx.scratch["original_query"] = f"what is this [IMAGE:{img}]"

    with pytest.raises(IngredientDetectionError):
        await tool.execute(ctx, image_path=str(img))

    assert img.exists()


async def test_successful_analysis_deletes_the_upload(uploads):
    img = uploads / "abc123.jpg"
    img.write_bytes(b"x")
    svc = ImageAnalysisService(_Detector(), None, upload_dir=str(uploads))
    tool = AnalyzeImageTool(svc)
    ctx = _ctx()
    ctx.scratch["original_query"] = f"what is this [IMAGE:{img}]"

    await tool.execute(ctx, image_path=str(img))

    assert not img.exists()
