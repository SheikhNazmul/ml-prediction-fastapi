from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=13,
        max_length=13,
        description="13 Wine dataset features in the documented order.",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: list[float]


class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    features: list[float]
    predicted_class: int
    class_name: str
    probabilities: list[float]
    created_at: datetime
