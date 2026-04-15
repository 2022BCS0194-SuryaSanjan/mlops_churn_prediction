from fastapi.testclient import TestClient
from api.main import app


payload = {
    'customer_id': 'CUST_0001',
    'monthly_charges': 95.0,
    'prev_monthly_charges': 70.0,
    'tickets_7d': 3,
    'tickets_30d': 8,
    'tickets_90d': 20,
    'avg_sentiment': -0.7,
    'category_billing': 5,
    'category_technical': 3,
    'category_general': 2,
    'days_since_last_ticket': 2,
    'contract_type': 'Month-to-month',
    'tenure_months': 3
}


client = TestClient(app)


def test_predict_endpoint():
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['customer_id'] == payload['customer_id']
    assert 'churn_prediction' in data
    assert 'churn_probability' in data
    assert 'risk_level' in data
