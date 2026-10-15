"""Relative routes mounted once by the shared ClinIQ entry point."""
from __future__ import annotations

import logging
import asyncio
import time
import uuid
import weakref
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import get_settings
from .preprocessing import ImageTooLargeError, InvalidImageError, decode_image
from .schemas import SkinHealthResponse, SkinPredictionResponse
from .service import ModelInferenceError, ModelUnavailableError, SkinPredictor, get_skin_predictor
from .upload import SkinRoute

router=APIRouter(tags=['Skin'],route_class=SkinRoute)
logger=logging.getLogger(__name__)
_INFERENCE_LIMITERS=weakref.WeakKeyDictionary()


def _request_limiter():
    loop=asyncio.get_running_loop()
    if loop not in _INFERENCE_LIMITERS:
        _INFERENCE_LIMITERS[loop]=asyncio.Semaphore(1)
    return _INFERENCE_LIMITERS[loop]


def get_skin_service() -> SkinPredictor:
    return get_skin_predictor()


@router.get('/health',response_model=SkinHealthResponse,response_model_by_alias=True)
def health_check(predictor: Annotated[SkinPredictor,Depends(get_skin_service)]):
    return SkinHealthResponse(status='ready' if predictor.is_ready else 'unavailable',
        model_available=predictor.is_ready,message='Skin model is ready.' if predictor.is_ready else predictor.unavailable_message,
        model_version=predictor.model_version,class_count=len(predictor.class_names))


@router.post('/predict',response_model=SkinPredictionResponse,response_model_by_alias=True)
async def predict_image(file: Annotated[UploadFile,File(description='Skin photograph in JPG, JPEG or PNG; prototype, not a medical diagnosis.')],
                        predictor: Annotated[SkinPredictor,Depends(get_skin_service)]):
    request_id=uuid.uuid4().hex
    started=time.perf_counter()
    size=0
    status=500
    abstained=None
    try:
        try:
            settings=get_settings()
        except ValueError as error:
            raise HTTPException(503,'Skin configuration is invalid.') from error
        extension=Path(file.filename or '').suffix.lower()
        if extension not in {'.jpg','.jpeg','.png'}:
            raise HTTPException(415,'Upload a JPG, JPEG, or PNG image.')
        chunks=[]
        while True:
            chunk=await file.read(min(65536,settings.max_upload_bytes+1-size))
            if not chunk:
                break
            size+=len(chunk)
            if size>settings.max_upload_bytes:
                raise HTTPException(413,'The uploaded image exceeds the configured size limit.')
            chunks.append(chunk)
        payload=b''.join(chunks)
        if not payload:
            raise HTTPException(400,'The uploaded image is empty.')
        jpeg=payload.startswith(b'\xff\xd8\xff')
        png=payload.startswith(b'\x89PNG\r\n\x1a\n')
        if not ((jpeg and extension in {'.jpg','.jpeg'}) or (png and extension=='.png')):
            raise HTTPException(415,'File contents must match a JPEG or PNG extension.')
        try:
            # Queue asynchronously; never occupy shared threadpool workers waiting.
            async with _request_limiter():
                await run_in_threadpool(decode_image,payload,settings.max_pixels)
                result=await run_in_threadpool(predictor.predict_bytes,payload)
        except ImageTooLargeError as error:
            raise HTTPException(413,str(error)) from error
        except InvalidImageError as error:
            raise HTTPException(400,'The uploaded image cannot be decoded.') from error
        except ModelUnavailableError as error:
            raise HTTPException(503,predictor.unavailable_message) from error
        except ModelInferenceError as error:
            raise HTTPException(500,'Skin model inference failed.') from error
        response=SkinPredictionResponse(**result)
        status=200
        abstained=response.abstained
        return response
    except HTTPException as error:
        status=error.status_code
        raise
    finally:
        await file.close()
        logger.info('Skin request requestId=%s bytes=%d status=%d latencyMs=%.3f modelVersion=%s abstained=%s',
                    request_id,size,status,(time.perf_counter()-started)*1000,predictor.model_version,abstained)
