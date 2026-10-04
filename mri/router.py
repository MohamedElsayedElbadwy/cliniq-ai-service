"""FastAPI router for the Brain MRI classification module.

The shared AI service's top-level ``main.py`` should include ``router``; this
module deliberately does not create a standalone FastAPI application.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from .config import ALLOWED_IMAGE_CONTENT_TYPES, ALLOWED_IMAGE_EXTENSIONS, get_settings
from .schemas import MRIHealthResponse, MRIPredictionResponse
from .service import InvalidImageError, ModelInferenceError, ModelUnavailableError, MRIPredictor, get_mri_predictor


router = APIRouter(tags=["MRI"])


def get_mri_service() -> MRIPredictor:
    """FastAPI dependency returning the cached MRI inference service."""
    return get_mri_predictor()


def _is_supported_upload(file: UploadFile) -> bool:
    suffix = ""
    if file.filename:
        suffix = file.filename.rsplit(".", maxsplit=1)[-1].lower()
        suffix = f".{suffix}" if suffix else ""
    return file.content_type in ALLOWED_IMAGE_CONTENT_TYPES or suffix in ALLOWED_IMAGE_EXTENSIONS


@router.get("/health", response_model=MRIHealthResponse, response_model_by_alias=True)
def health_check(predictor: Annotated[MRIPredictor, Depends(get_mri_service)]) -> MRIHealthResponse:
    """Report the cached MRI predictor's current availability."""
    if predictor.is_ready:
        return MRIHealthResponse(status="ready", model_available=True, message="MRI model weights are available.")
    return MRIHealthResponse(
        status="unavailable",
        model_available=False,
        message=predictor.unavailable_message,
    )


@router.post("/predict", response_model=MRIPredictionResponse, response_model_by_alias=True)
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

    try:
        result = await run_in_threadpool(predictor.predict_bytes, payload)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except InvalidImageError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except ModelInferenceError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MRI model inference failed.") from error
    return MRIPredictionResponse(
        prediction=result["prediction"],
        class_index=result["class_index"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )
