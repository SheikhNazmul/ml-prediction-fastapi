from fastapi import FastAPI

from app.model import CLASS_NAMES, FEATURE_NAMES, MODEL
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="ML Prediction API",
    description=(
        "A production-style FastAPI service demonstrating machine-learning "
        "inference, validation, and interactive API documentation."
    ),
    version="1.0.0",
)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {"message": "ML Prediction API is running", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "wine-classifier"}


@app.get("/features", tags=["Model"])
def features() -> dict[str, object]:
    return {"count": len(FEATURE_NAMES), "order": FEATURE_NAMES}


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = int(MODEL.predict([request.features])[0])
    probabilities = MODEL.predict_proba([request.features])[0].tolist()
    return PredictionResponse(
        predicted_class=prediction,
        class_name=CLASS_NAMES[prediction],
        probabilities=[round(value, 6) for value in probabilities],
    )
