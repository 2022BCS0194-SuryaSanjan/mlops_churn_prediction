import pytest
import joblib
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.features import engineer_features




@pytest.fixture
def model():
    if not os.path.exists('models/churn_model.pkl'):
        pytest.skip('Model not trained yet')
    return joblib.load('models/churn_model.pkl')




def test_model_loads(model):
    assert model is not None




def test_prediction_output_shape(model):
    df = pd.DataFrame([{
        'monthly_charges': 80.0, 'prev_monthly_charges': 60.0,
        'tickets_7d': 2, 'tickets_30d': 6, 'tickets_90d': 15,
        'avg_sentiment': -0.4, 'category_billing': 3,
        'category_technical': 2, 'category_general': 1,
        'days_since_last_ticket': 5, 'contract_type': 'Month-to-month',
        'tenure_months': 6
    }])
    df = engineer_features(df)
    cols = ['monthly_charges','charge_change','tickets_7d','tickets_30d',
            'tickets_90d','avg_sentiment','ticket_acceleration','billing_ratio',
            'days_since_last_ticket','tenure_months','contract_type','sentiment_bucket']
    pred = model.predict(df[cols])
    prob = model.predict_proba(df[cols])
    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0
