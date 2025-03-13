from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict
from src.models.train import FraudDetectionModel
from src.data.preprocessing import DataPreprocessor

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_type: str

# Initialize models and preprocessor
model = FraudDetectionModel()
preprocessor = DataPreprocessor()

@app.post("/predict/isolation-forest", response_model=PredictionResponse)
async def predict_isolation_forest(transaction: Transaction):
    try:
        # Convert input to numpy array and reshape for single prediction
        features = np.array(transaction.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict_isolation_forest(features)[0]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=1.0,  # Isolation Forest doesn't provide probabilities
            model_type="isolation_forest"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/xgboost", response_model=PredictionResponse)
async def predict_xgboost(transaction: Transaction):
    try:
        # Convert input to numpy array and reshape for single prediction
        features = np.array(transaction.features).reshape(1, -1)
        
        # Make prediction and get probability
        prediction = model.predict_xgboost(features)[0]
        probability = model.predict_xgboost_proba(features)[0][1]  # Probability of fraud
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_type="xgboost"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}