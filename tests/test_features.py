import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.features import engineer_features




def make_sample_df():
    return pd.DataFrame({
        'monthly_charges': [80.0],
        'prev_monthly_charges': [60.0],
        'tickets_7d': [2],
        'tickets_30d': [6],
        'tickets_90d': [15],
        'avg_sentiment': [-0.4],
        'category_billing': [3],
        'category_technical': [2],
        'category_general': [1],
        'days_since_last_ticket': [5],
        'contract_type': ['Month-to-month'],
        'tenure_months': [6]
    })




def test_charge_change():
    df = make_sample_df()
    result = engineer_features(df)
    assert 'charge_change' in result.columns
    assert result['charge_change'].iloc[0] == pytest.approx(20.0)




def test_ticket_acceleration():
    df = make_sample_df()
    result = engineer_features(df)
    assert 'ticket_acceleration' in result.columns




def test_billing_ratio_between_0_and_1():
    df = make_sample_df()
    result = engineer_features(df)
    assert 0 <= result['billing_ratio'].iloc[0] <= 1




def test_sentiment_bucket_values():
    df = make_sample_df()
    result = engineer_features(df)
    assert result['sentiment_bucket'].iloc[0] in ['negative', 'neutral', 'positive']




def test_no_nulls_after_engineering():
    df = make_sample_df()
    result = engineer_features(df)
    assert result.isnull().sum().sum() == 0
