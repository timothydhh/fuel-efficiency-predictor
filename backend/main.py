from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # boleh semua (untuk development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model_rf.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

print("MODEL PATH:", model_path)
print("SCALER PATH:", scaler_path)

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


class VehicleInput(BaseModel):
    cylinders: float
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: float
    origin_1: int
    origin_2: int
    origin_3: int


@app.get("/")
def home():
    return {"message": "Fuel Efficiency API Running"}

@app.post("/predict")
def predict(data: VehicleInput):
    try:
        features = [
            data.cylinders,
            data.displacement,
            data.horsepower,
            data.weight,
            data.acceleration,
            data.model_year,
            data.origin_1,
            data.origin_2,
            data.origin_3
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        prediction_kml = float(prediction) / 3.78541

        return {
            "predicted_km_per_liter": round(prediction_kml, 2)
        }

    except Exception as e:
        return {"error": str(e)}