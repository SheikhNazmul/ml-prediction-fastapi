# ML Prediction FastAPI

A portfolio-ready machine-learning REST API built with **FastAPI** and **scikit-learn**. The project demonstrates model training, request validation, inference, API documentation, health checks, and Docker-ready deployment.

## What it demonstrates

- FastAPI REST API design
- Pydantic request/response validation
- Reproducible scikit-learn pipeline
- Feature metadata endpoint
- Prediction probabilities
- Interactive Swagger UI at `/docs`
- Health monitoring endpoint
- Docker deployment configuration
- Clean separation of API, schemas, and model code

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status and documentation link |
| GET | `/health` | Health check |
| GET | `/features` | Required feature order |
| POST | `/predict` | Run an ML prediction |
| GET | `/docs` | Interactive Swagger documentation |

## Model

The demo uses the public scikit-learn Wine dataset and a `StandardScaler + LogisticRegression` pipeline. The model is trained reproducibly when the application starts, so no binary model artifact is required in the repository.

The prediction request accepts exactly 13 numeric features in this order:

`alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline`

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker build -t ml-prediction-fastapi .
docker run -p 8000:8000 ml-prediction-fastapi
```

## Portfolio

This repository is part of Sheikh Nazmul Islam's AI/ML portfolio and demonstrates practical Python, machine learning, REST API, and deployment skills.
