"""FastAPI router for the Brain MRI classification module.

The shared AI service's top-level ``main.py`` should include ``router``; this
module deliberately does not create a standalone FastAPI application.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from .config import ALLOWED_IMAGE_CONTENT_TYPES, ALLOWED_IMAGE_EXTENSIONS, get_settings
from .schemas import MRIHealthResponse, MRIPredictionResponse
from .service import MRIPredictor, get_mri_predictor


router = APIRouter(prefix="/mri", tags=["MRI"])


def get_mri_service() -> MRIPredictor:
    """FastAPI dependency returning the cached MRI inference service."""
    return get_mri_predictor()


def _is_supported_upload(file: UploadFile) -> bool:
    suffix = ""
    if file.filename:
        suffix = file.filename.rsplit(".", maxsplit=1)[-1].lower()
        suffix = f".{suffix}" if suffix else ""
    return file.content_type in ALLOWED_IMAGE_CONTENT_TYPES or suffix in ALLOWED_IMAGE_EXTENSIONS


@router.get("/health", response_model=MRIHealthResponse)
def health_check() -> MRIHealthResponse:
    """Report whether model weights are available without loading them."""
    settings = get_settings()
    if settings.model_path:
        return MRIHealthResponse(status="ready", model_available=True, message="MRI model weights are available.")
    return MRIHealthResponse(
        status="unavailable",
        model_available=False,
        message="MRI model weights are unavailable. Configure CLINIQ_MRI_MODEL_PATH before inference.",
    )


@router.post("/predict", response_model=MRIPredictionResponse)
async def predict_image(
    file: Annotated[UploadFile, File(description="MRI image in JPG, JPEG, or PNG format")],
    predictor: Annotated[MRIPredictor, Depends(get_mri_service)],
) -> MRIPredictionResponse:
    """Classify one uploaded MRI image using the existing trained model."""
    if not _is_supported_upload(file):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload a JPG, JPEG, or PNG image.")

    max_upload_size = get_settings().max_upload_size_bytes
    # Read at most one byte beyond the limit so a large upload is rejected
    # without unnecessarily retaining its full contents in application memory.
    payload = await file.read(max_upload_size + 1)
    if not payload:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The uploaded image is empty.")
    if len(payload) > max_upload_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="The uploaded image exceeds the 10 MB limit.")

    result = predictor.predict_bytes(payload)
    if not result["success"]:
        error = result["error"]
        unavailable = "weights are unavailable" in error.lower()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE if unavailable else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error,
        )
    return MRIPredictionResponse(
        prediction=result["prediction"],
        class_index=result["class_index"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )
