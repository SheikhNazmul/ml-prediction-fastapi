from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.db_models import PredictionHistory
from app.logger import logger
from app.model import CLASS_NAMES, FEATURE_NAMES, MODEL
from app.schemas import (
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ML Prediction API")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables are ready")
    except SQLAlchemyError:
        logger.exception("Database initialization failed")
        raise

    yield

    logger.info("Shutting down ML Prediction API")


app = FastAPI(
    title="ML Prediction API",
    description=(
        "A production-style FastAPI service demonstrating machine-learning "
        "inference, PostgreSQL prediction history, validation, error handling, "
        "logging, and interactive API documentation."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "Validation error | method=%s path=%s errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception(
        "Unhandled error | method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {"message": "ML Prediction API is running", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health(db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "model": "wine-classifier", "database": "connected"}
    except SQLAlchemyError as exc:
        logger.exception("Health check failed: database unavailable")
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc


@app.get("/features", tags=["Model"])
def features() -> dict[str, object]:
    return {"count": len(FEATURE_NAMES), "order": FEATURE_NAMES}


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    logger.info("Prediction request received")

    try:
        prediction = int(MODEL.predict([request.features])[0])
        probabilities = MODEL.predict_proba([request.features])[0].tolist()
        class_name = CLASS_NAMES[prediction]
        rounded_probabilities = [round(value, 6) for value in probabilities]

        history = PredictionHistory(
            features=request.features,
            predicted_class=prediction,
            class_name=class_name,
            probabilities=rounded_probabilities,
        )
        db.add(history)
        db.commit()

        logger.info(
            "Prediction completed | class=%s class_name=%s history_id=%s",
            prediction,
            class_name,
            history.id,
        )

        return PredictionResponse(
            predicted_class=prediction,
            class_name=class_name,
            probabilities=rounded_probabilities,
        )

    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to save prediction history")
        raise HTTPException(
            status_code=503,
            detail="Prediction succeeded but history could not be saved",
        ) from exc
    except (ValueError, IndexError) as exc:
        logger.exception("Prediction processing failed")
        raise HTTPException(
            status_code=400,
            detail="Unable to process the prediction input",
        ) from exc


@app.get(
    "/predictions",
    response_model=list[PredictionHistoryResponse],
    tags=["History"],
)
def prediction_history(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[PredictionHistoryResponse]:
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100",
        )

    try:
        result = db.execute(
            select(PredictionHistory)
            .order_by(PredictionHistory.created_at.desc())
            .limit(limit)
        )
        return [
            PredictionHistoryResponse.model_validate(item, from_attributes=True)
            for item in result.scalars().all()
        ]
    except SQLAlchemyError as exc:
        logger.exception("Failed to fetch prediction history")
        raise HTTPException(
            status_code=503,
            detail="Prediction history is temporarily unavailable",
        ) from exc
