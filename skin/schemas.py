"""Dataset-class predictions for decision support, not a medical diagnosis."""
from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER='Decision-support prototype. Not a medical diagnosis. Consult a qualified clinician.'


class SkinResponseModel(BaseModel):
    model_config=ConfigDict(populate_by_name=True)


class SkinErrorResponse(SkinResponseModel):
    detail: str


class TopPrediction(SkinResponseModel):
    label: str
    probability: float=Field(ge=0,le=1)


class SkinPredictionResponse(SkinResponseModel):
    predicted_class: str | None=Field(alias='predictedClass')
    confidence: float=Field(ge=0,le=1,description='Model class score, NOT diagnostic correctness. MAKE zero-shot and pretrained softmax scores are uncalibrated; MAKE scores depend on six fixed descriptions.')
    abstained: bool
    abstain_reason: str | None=Field(alias='abstainReason')
    top_predictions: list[TopPrediction]=Field(alias='topPredictions')
    probabilities: dict[str,float]=Field(description='All native class or fixed zero-shot candidate scores sum to one; topPredictions is a subset. Labels depend on modelVersion; scores do not guarantee correctness.')
    model_version: str=Field(alias='modelVersion')
    inference_ms: float=Field(alias='inferenceMs',ge=0)
    disclaimer: str=DISCLAIMER


class SkinHealthResponse(SkinResponseModel):
    status: str
    model_available: bool=Field(alias='modelAvailable')
    message: str
    model_version: str | None=Field(alias='modelVersion',default=None)
    class_count: int=Field(alias='classCount',default=0)
