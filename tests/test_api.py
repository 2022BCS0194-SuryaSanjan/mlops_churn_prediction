import requests, json


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


response = requests.post('http://127.0.0.1:8000/predict', json=payload)
print(json.dumps(response.json(), indent=2))
