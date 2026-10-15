"""Dataset-class predictions for decision support, not a medical diagnosis."""
from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER='Decision-support prototype. Not a medical diagnosis. Consult a qualified clinician.'


class SkinResponseModel(BaseModel):
    model_config=ConfigDict(populate_by_name=True)


class TopPrediction(SkinResponseModel):
    label: str
    probability: float=Field(ge=0,le=1)


class SkinPredictionResponse(SkinResponseModel):
    predicted_class: str | None=Field(alias='predictedClass')
    confidence: float=Field(ge=0,le=1,description='Calibrated model class probability, NOT the probability that a diagnosis is correct.')
    abstained: bool
    abstain_reason: str | None=Field(alias='abstainReason')
    top_predictions: list[TopPrediction]=Field(alias='topPredictions')
    probabilities: dict[str,float]=Field(description='All class probabilities sum to one; topPredictions is a subset.')
    model_version: str=Field(alias='modelVersion')
    inference_ms: float=Field(alias='inferenceMs',ge=0)
    disclaimer: str=DISCLAIMER


class SkinHealthResponse(SkinResponseModel):
    status: str
    model_available: bool=Field(alias='modelAvailable')
    message: str
    model_version: str | None=Field(alias='modelVersion',default=None)
    class_count: int=Field(alias='classCount',default=0)
