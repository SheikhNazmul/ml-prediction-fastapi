from pydantic import BaseModel, Field


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
