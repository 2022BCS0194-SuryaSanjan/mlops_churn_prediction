import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


np.random.seed(42)
n = 2000


customer_ids = [f'CUST_{i:04d}' for i in range(n)]


data = {
    'customer_id': customer_ids,
    'monthly_charges': np.round(np.random.uniform(20, 120, n), 2),
    'prev_monthly_charges': np.round(np.random.uniform(20, 120, n), 2),
    'tickets_7d': np.random.poisson(1, n),
    'tickets_30d': np.random.poisson(3, n),
    'tickets_90d': np.random.poisson(8, n),
    'avg_sentiment': np.round(np.random.uniform(-1, 1, n), 3),
    'category_billing': np.random.poisson(2, n),
    'category_technical': np.random.poisson(3, n),
    'category_general': np.random.poisson(1, n),
    'days_since_last_ticket': np.random.randint(0, 90, n),
    'contract_type': np.random.choice(['Month-to-month','One year','Two year'], n),
    'tenure_months': np.random.randint(1, 72, n),
}


df = pd.DataFrame(data)


# Create churn label (rule-based for ground truth)
df['churn'] = (
    (df['tickets_30d'] > 5) |
    (df['avg_sentiment'] < -0.5) |
    ((df['monthly_charges'] - df['prev_monthly_charges']) > 20) |
    (df['contract_type'] == 'Month-to-month')
).astype(int)


# Add noise
flip = np.random.choice([True, False], n, p=[0.05, 0.95])
df.loc[flip, 'churn'] = 1 - df.loc[flip, 'churn']


df.to_csv('data/raw/customers.csv', index=False)
print(f'Dataset created: {n} rows, churn rate: {df.churn.mean():.2%}')
