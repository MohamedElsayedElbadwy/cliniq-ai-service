"""Request and response schemas for MRI image classification endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MRIResponseModel(BaseModel):
    """Base response model that accepts Python names and emits API aliases."""

    model_config = ConfigDict(populate_by_name=True)


class MRIPredictionResponse(MRIResponseModel):
    """A model classification result; it is not a medical diagnosis."""

    prediction: str = Field(description="Predicted dataset class label.")
    class_index: int = Field(alias="classIndex", ge=0, description="Index of the predicted class.")
    confidence: float = Field(ge=0.0, le=1.0, description="Selected softmax model output.")
    probabilities: dict[str, float] = Field(description="Softmax score for every supported class.")


class MRIHealthResponse(MRIResponseModel):
    """MRI module availability, without loading or exposing model internals."""

    status: str
    model_available: bool = Field(alias="modelAvailable")
    message: str
