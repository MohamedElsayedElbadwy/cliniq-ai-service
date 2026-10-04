"""Request and response schemas for MRI image classification endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MRIPredictionResponse(BaseModel):
    """A model classification result; it is not a medical diagnosis."""

    prediction: str = Field(description="Predicted dataset class label.")
    class_index: int = Field(ge=0, description="Index of the predicted class.")
    confidence: float = Field(ge=0.0, le=1.0, description="Selected softmax model output.")
    probabilities: dict[str, float] = Field(description="Softmax score for every supported class.")


class MRIHealthResponse(BaseModel):
    """MRI module availability, without loading or exposing model internals."""

    status: str
    model_available: bool
    message: str
