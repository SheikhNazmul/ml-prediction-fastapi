# ML Prediction FastAPI

A production-style machine-learning REST API built with **FastAPI**, **scikit-learn**, and **PostgreSQL**. The project demonstrates ML inference, request validation, prediction persistence, error handling, application logging, health checks, API documentation, and Docker deployment.

## What it demonstrates

- FastAPI REST API design
- Pydantic request/response validation
- Reproducible scikit-learn pipeline
- `StandardScaler + LogisticRegression`
- Prediction probabilities
- PostgreSQL prediction history with SQLAlchemy
- Database health checks
- Structured application logging
- Validation and global exception handling
- Interactive Swagger UI at `/docs`
- Docker and Docker Compose support
- Clean separation of API, schemas, model, database, and logging code

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | API + PostgreSQL health check |
| GET | `/features` | Required feature order |
| POST | `/predict` | Run prediction and save it to PostgreSQL |
| GET | `/predictions?limit=20` | Retrieve recent prediction history |
| GET | `/docs` | Interactive Swagger documentation |

## PostgreSQL prediction history

Every successful prediction is stored in the `prediction_history` table with input features, predicted class, class name, probabilities, and UTC timestamp.

The API uses SQLAlchemy 2.x and the Psycopg PostgreSQL driver.

## Error handling

The API handles invalid request payloads with structured 422 responses, invalid prediction inputs with 400 responses, database failures with 503 responses, and unexpected application errors with a generic 500 response. Database transactions are rolled back when history persistence fails.

## Logging

Application events are logged with timestamps and log levels, including startup/shutdown, database initialization, prediction requests, successful predictions, validation failures, database errors, and unexpected exceptions.

Set `LOG_LEVEL` through the environment, for example `INFO` or `DEBUG`.

## Run with Docker Compose

Make sure Docker Desktop is running, then:

~~~bash
docker compose up --build
~~~

Open:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Stop:

~~~bash
docker compose down
~~~

## Run with an existing PostgreSQL database

Create a PostgreSQL database named `ml_prediction`.

Copy `.env.example` to `.env` and update:

~~~env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/ml_prediction
LOG_LEVEL=INFO
~~~

Install and run:

~~~bash
python -m venv .venv
pip install -r requirements.txt
uvicorn app.main:app --reload
~~~

On Windows PowerShell, activate the environment with:

~~~powershell
.venv\\Scripts\\Activate.ps1
~~~

The application creates the `prediction_history` table automatically on startup.

## Example prediction

POST `/predict` with exactly 13 numeric values:

~~~json
{
  "features": [
    13.2, 1.78, 2.14, 11.2, 100,
    2.65, 2.76, 0.26, 1.28, 4.38,
    1.05, 3.4, 1050
  ]
}
~~~

Then call `GET /predictions` to see the saved history.

## Portfolio

This repository is part of Sheikh Nazmul Islam's AI/ML portfolio and demonstrates practical Python, machine learning, REST API, PostgreSQL, backend engineering, error handling, logging, and containerization.
