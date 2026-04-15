from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.features import engineer_features


app = FastAPI(title='Churn Prediction API', version='1.0.0')


# Load model at startup
MODEL_PATH = 'models/churn_model.pkl'
model = None


@app.on_event('startup')
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print('Model loaded successfully')
    else:
        print(f'WARNING: Model not found at {MODEL_PATH}')




class CustomerData(BaseModel):
    customer_id: str
    monthly_charges: float
    prev_monthly_charges: float
    tickets_7d: int
    tickets_30d: int
    tickets_90d: int
    avg_sentiment: float
    category_billing: int
    category_technical: int
    category_general: int
    days_since_last_ticket: int
    contract_type: str
    tenure_months: int




class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: int
    churn_probability: float
    risk_level: str




@app.get('/')
def root():
    return {'message': 'Churn Prediction API is running', 'model_loaded': model is not None}




@app.post('/predict', response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')


    # Convert to DataFrame
    data = customer.dict()
    cid = data.pop('customer_id')
    df = pd.DataFrame([data])


    # Engineer features
    df = engineer_features(df)


    # Select feature columns
    feature_cols = [
        'monthly_charges', 'charge_change', 'tickets_7d', 'tickets_30d',
        'tickets_90d', 'avg_sentiment', 'ticket_acceleration', 'billing_ratio',
        'days_since_last_ticket', 'tenure_months', 'contract_type', 'sentiment_bucket'
    ]
    X = df[feature_cols]


    # Predict
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])


    # Risk classification
    if probability >= 0.7:
        risk = 'HIGH'
    elif probability >= 0.4:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'


    return PredictionResponse(
        customer_id=cid,
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk
    )




@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}
