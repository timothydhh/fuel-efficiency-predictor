# Fuel Efficiency Predictor

A machine learning web application that predicts vehicle fuel efficiency (km/g) from engine and vehicle specifications. Built as a BINUS University COMP6577 final project using the UCI Auto MPG dataset, with two trained models (Linear Regression and Random Forest) served via a FastAPI backend and a plain HTML/JS frontend.

## Tech Stack

- **Python** — data processing, model training
- **Scikit-learn** — LinearRegression, RandomForestRegressor, StandardScaler
- **FastAPI** — REST API backend
- **HTML / CSS / JavaScript** — frontend UI

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn numpy pandas scikit-learn matplotlib seaborn
   ```

2. Start the backend:
   ```bash
   cd backend && uvicorn main:app --reload
   ```

3. Open `frontend/index.html` in a browser.

## How to Retrain the Model

```bash
cd dataset && jupyter notebook data_processing.ipynb
```

Run all cells in order. This regenerates `model_lr.pkl`, `model_rf.pkl`, and `scaler.pkl`.

## Deployment

TODO: add Render URL here

- Backend: deploy the `backend/` folder as a FastAPI service on Render.
- Frontend: host `frontend/index.html` on GitHub Pages or Netlify, then update `API_URL` in `index.html`.

## Dataset

UCI Auto MPG Dataset — 392 samples of vehicle specifications from 1970–1982.  
Source: <https://archive.ics.uci.edu/dataset/9/auto+mpg>
