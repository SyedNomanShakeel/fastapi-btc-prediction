# BTC Prediction FastAPI

A FastAPI-based API for user management and BTC price prediction using a pre-trained ML model.

## Features

- User signup and login (SQLite)
- Predict BTC price based on market features
- CORS enabled

## Endpoints

- `POST /signup` — Register a new user
- `POST /login` — Login user
- `POST /predict` — Predict BTC price

## Deployment

1. Place your `rf_random_model.pkl` and `scaler_model.pkl` in the `app/` directory.
2. Deploy on [Render](https://render.com/) using the included `render.yaml`.

## Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```