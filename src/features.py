import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw customer data."""
    df = df.copy()


    # Feature: Change in monthly charges
    df['charge_change'] = df['monthly_charges'] - df['prev_monthly_charges']


    # Feature: Ticket acceleration (7d vs 30d rate)
    df['ticket_acceleration'] = df['tickets_7d'] - (df['tickets_30d'] / 4)


    # Feature: Total ticket count
    df['total_tickets'] = df['tickets_90d']


    # Feature: Billing ticket ratio
    total_cat = df['category_billing'] + df['category_technical'] + df['category_general'] + 1
    df['billing_ratio'] = df['category_billing'] / total_cat


    # Feature: Sentiment bucket
    df['sentiment_bucket'] = pd.cut(
        df['avg_sentiment'],
        bins=[-1.1, -0.3, 0.3, 1.1],
        labels=['negative', 'neutral', 'positive']
    ).astype(str)


    return df




def build_preprocessor():
    """Build sklearn preprocessing pipeline."""
    numeric_features = [
        'monthly_charges', 'charge_change', 'tickets_7d', 'tickets_30d',
        'tickets_90d', 'avg_sentiment', 'ticket_acceleration',
        'billing_ratio', 'days_since_last_ticket', 'tenure_months'
    ]
    categorical_features = ['contract_type', 'sentiment_bucket']


    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor




def prepare_data(input_path: str, output_dir: str):
    """Full data preparation pipeline."""
    from sklearn.model_selection import train_test_split


    os.makedirs(output_dir, exist_ok=True)


    # Load and engineer features
    df = pd.read_csv(input_path)
    df = engineer_features(df)


    # Define features and target
    feature_cols = [
        'monthly_charges', 'charge_change', 'tickets_7d', 'tickets_30d',
        'tickets_90d', 'avg_sentiment', 'ticket_acceleration', 'billing_ratio',
        'days_since_last_ticket', 'tenure_months', 'contract_type', 'sentiment_bucket'
    ]
    X = df[feature_cols]
    y = df['churn']


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Save splits
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)


    # Fit and save preprocessor on training data only
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, f'{output_dir}/preprocessor.pkl')


    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')
    print(f'Churn rate (train): {y_train.mean():.2%}')
    return X_train, X_test, y_train, y_test




if __name__ == '__main__':
    prepare_data('data/raw/customers.csv', 'data/splits')
