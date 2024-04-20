# Used Car Price Prediction
This project presents a simple ML project, from problem definition to inference.

The problem uses a Kaggle dataset [https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data]


Run the API server:
```
uvicorn app:app --reload
```

Example:
```
curl -X POST "http://127.0.0.1:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"Present_Price\":10.0,\"Year\":2017,\"Kms_Driven\":45000,\"Fuel_Type\":\"Petrol\",\"Transmission\":\"Manual\"}"
```