"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: API prototype
-----------------------------------------------------------------------------------------------------------------------
Process:
1. Reads model artifact path from config toml
2. Generate app
3. Create fake records to test the app
4. Run the app and go to UI http://127.0.0.1:8000 
5. Go to POST/predict, edit the values and EXECUTE to generate the prediction

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
import random
import Global
import os


config = Global.load_config()
model_name = config["model"]["algo"]["best_model"]
path = Global.get_project_path(config["model"]["save"]["path"], 0)
data_path = Global.get_project_path(config["model"]["data_prep"]["path"], 0)
model_filename = f"{model_name}_best_model.joblib"
MODEL_PATH = os.path.join(path, model_filename)


try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model not loaded. Please check MODEL_PATH. ({e})")

app = FastAPI(
    title="PG Technical Assessment Food Delivery Time Prediction API",
    description="An API that predicts delivery time based on distance, weather, traffic, etc.",
    version="1.0.0"
)


class DeliveryFeatures(BaseModel):
    Distance_km: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str
    Preparation_Time_min: float
    Courier_Experience_yrs: float


@app.get("/")
def home():
    return {"message": "PG Technical Assessment Food Delivery Time Prediction API"}


@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "detail": "Model not loaded. Check MODEL_PATH."}
    return {"status": "ok"}

@app.post("/predict")
def predict(features: DeliveryFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check MODEL_PATH."}

    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]

    return {
        "input_data": features.dict(),
        "predicted_delivery_time_min": round(float(prediction), 2)
    }

@app.get("/sample-data")
def generate_sample_data(n: int = 5):
    """
    _____________________________________________________________________________________________________________
    Generates fake delivery feature data to test the /predict endpoint
    _____________________________________________________________________________________________________________
    """
    np.random.seed(10)
    weather_options = ["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
    traffic_options = ["Low", "Medium", "High"]
    time_options = ["Morning", "Afternoon", "Evening", "Night"]
    vehicle_options = ["Bike", "Scooter", "Car"]

    data = []
    for _ in range(n):
        entry = {
            "Distance_km": round(np.random.uniform(0.5, 15), 2),
            "Weather": random.choice(weather_options),
            "Traffic_Level": random.choice(traffic_options),
            "Time_of_Day": random.choice(time_options),
            "Vehicle_Type": random.choice(vehicle_options),
            "Preparation_Time_min": round(np.random.uniform(5, 30), 1),
            "Courier_Experience_yrs": round(np.random.uniform(0, 10), 1)
        }
        data.append(entry)

    return {"sample_records": data}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
