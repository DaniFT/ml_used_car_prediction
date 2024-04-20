from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pickle
import numpy as np


class CarData(BaseModel):
    Present_Price: float
    Year: int
    Kms_Driven: int
    Fuel_Type: str
    Transmission: str

    @field_validator('Fuel_Type')
    def validate_fuel_type(cls: Any, v: str) -> str:
        valid_types = {'Petrol', 'Diesel', 'CNG'}
        if v not in valid_types:
            raise ValueError('Fuel_Type must be either Petrol, Diesel, or CNG')
        return v

    @field_validator('Transmission')
    def validate_transmission(cls: Any, v: str) -> str:
        if v not in {'Manual', 'Automatic'}:
            raise ValueError('Transmission must be either Manual or Automatic')
        return v


with open('models/random_forest_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()


@app.post('/predict')
async def predict(car: CarData) -> Dict[str, float]:
    try:
        # Map Fuel_Type and Transmission to the encoded format used in training
        fuel_type_encoded = [1 if car.Fuel_Type == 'Petrol' else 0,
                             1 if car.Fuel_Type == 'Diesel' else 0,
                             1 if car.Fuel_Type == 'CNG' else 0]
        transmission_encoded = [1 if car.Transmission == 'Manual' else 0,
                                1 if car.Transmission == 'Automatic' else 0]

        # Concatenate features in the expected order
        features = np.array([[car.Present_Price, car.Year, car.Kms_Driven] +
                             fuel_type_encoded + transmission_encoded])

        # Predict the price
        prediction = model.predict(features)
        return {"predicted_selling_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)