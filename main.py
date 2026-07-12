"""
Car Price Prediction — FastAPI Backend
=======================================
Production-ready REST API that performs inference using the pre-trained
Random Forest pipeline. No retraining happens here.

How to Run (Uvicorn)
--------------------
1. Install dependencies:
       pip install -r requirements.txt

2. Development mode (auto-reload on code changes):
       uvicorn main:app --reload

3. Production mode (multi-worker, bound to all interfaces):
       uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

   The server will start at: http://127.0.0.1:8000

How to Test (Swagger UI)
------------------------
1. Start the server with any command above.
2. Open your browser and go to:
       http://127.0.0.1:8000/docs         (Swagger UI — interactive)
       http://127.0.0.1:8000/redoc        (ReDoc — read-only docs)
3. In Swagger UI:
       - Expand the POST /predict endpoint
       - Click "Try it out"
       - Paste a JSON body, e.g.:
         {
           "make": "toyota",
           "model": "camry",
           "engine_fuel_type": "regular_unleaded",
           "engine_hp": 203.0,
           "engine_cylinders": 4.0,
           "transmission_type": "automatic",
           "driven_wheels": "front_wheel_drive",
           "number_of_doors": 4.0,
           "vehicle_size": "midsize",
           "vehicle_style": "sedan",
           "city_mpg": 28,
           "year": 2015
         }
       - Click "Execute" and view the predicted_price in the response.
"""

import pathlib
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# -----------------------------
# 1. Load trained model
# -----------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_rf_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# 2. Create FastAPI app
# -----------------------------

app= FastAPI(title="Car Price Prediction API", 
             description="Predicts MSRP of a car using a trained Random Forest model.", 
             version="1.0"
)

# --------------------------------------------------
# 3. Input Schema
# --------------------------------------------------

class CarFeatures(BaseModel):

    make: str = Field(...)
    model: str = Field(...)

    engine_fuel_type: str = Field(...)

    engine_hp: float = Field(..., gt=0, le=1200)

    engine_cylinders: float = Field(..., ge=0, le=16)

    transmission_type: str = Field(...)

    driven_wheels: str = Field(...)

    number_of_doors: float = Field(..., ge=2, le=4)

    vehicle_size: str = Field(...)

    vehicle_style: str = Field(...)

    city_mpg: int = Field(..., gt=1, le=150)

    year: int = Field(..., ge=1990, le=2017)


# ------------------------
# 4. Validators
# ------------------------

@field_validator("number_of_doors")
@classmethod
def validate_doors(cls, value):

    if value not in (2, 4, 2.0, 4.0):
        raise ValueError("Number of doors must be either 2 or 4.")

    return value

@field_validator(
        "make",
        "model",
        "engine_fuel_type",
        "transmission_type",
        "driven_wheels",
        "vehicle_size",
        "vehicle_style",
        mode="before",
)
@classmethod
def normalize_strings(cls, value):

    return str(value).lower().replace(" ", "_")

# --------------------------------------------------
# 5. Feature Engineering
# --------------------------------------------------

def prepare_input(car: CarFeatures):

    age = 2017 - car.year

    hp_per_cylinder = car.engine_hp / max(car.engine_cylinders, 1)

    return pd.DataFrame([{
        "make": car.make,
        "model": car.model,
        "engine_fuel_type": car.engine_fuel_type,
        "engine_hp": car.engine_hp,
        "engine_cylinders": car.engine_cylinders,
        "transmission_type": car.transmission_type,
        "driven_wheels": car.driven_wheels,
        "number_of_doors": car.number_of_doors,
        "vehicle_size": car.vehicle_size,
        "vehicle_style": car.vehicle_style,
        "city_mpg": car.city_mpg,
        "age": age,
        "hp_per_cylinder": hp_per_cylinder
    }])

# -----------------------------
# 6. Home route
# -----------------------------

@app.get("/")
def home():
    return {"message": "Welcome to the Car Price Prediction API!"}

# -----------------------------
# 6.1 Health Check
# -----------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Used by Streamlit, Docker, Render, Railway, etc.
    to verify that the API is running.
    """
    return {
        "status": "healthy"
    }

# -----------------------------
# 7. Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict_price(car: CarFeatures):

    try:

        input_df = prepare_input(car)

        prediction = model.predict(input_df)[0]

        prediction = max(0, float(prediction))

        return {
            "predicted_price": round(prediction, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

