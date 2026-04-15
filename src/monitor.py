import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')




def compute_psi(expected, actual, bins=10):
    """Population Stability Index - detects feature drift."""
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()


    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)


    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)


    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 0.0001, actual_pct)


    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)




def detect_drift(reference_path, current_path, output_path='models/drift_report.json'):
    """Compare reference data vs current data for feature drift."""
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)


    numeric_cols = ['monthly_charges', 'tickets_7d', 'tickets_30d',
                    'tickets_90d', 'avg_sentiment', 'tenure_months']


    report = {
        'timestamp': datetime.now().isoformat(),
        'reference_rows': len(ref),
        'current_rows': len(cur),
        'feature_drift': {},
        'overall_status': 'OK'
    }


    drift_detected = False
    for col in numeric_cols:
        if col in ref.columns and col in cur.columns:
            psi = compute_psi(ref[col], cur[col])
            status = 'DRIFT' if psi > 0.2 else ('WARNING' if psi > 0.1 else 'OK')
            report['feature_drift'][col] = {'psi': round(psi, 4), 'status': status}
            if status == 'DRIFT':
                drift_detected = True
                print(f'  DRIFT detected in {col}: PSI={psi:.4f}')
            else:
                print(f'  {col}: PSI={psi:.4f} ({status})')


    if drift_detected:
        report['overall_status'] = 'DRIFT_DETECTED'
        print('\nACTION: Retraining recommended!')
    else:
        print('\nAll features stable. No retraining needed.')


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


    return report




if __name__ == '__main__':
    # Simulate new data with slight drift
    import shutil
    shutil.copy('data/raw/customers.csv', 'data/raw/current_batch.csv')


    # Add synthetic drift to simulate production data changes
    df = pd.read_csv('data/raw/current_batch.csv')
    df['monthly_charges'] = df['monthly_charges'] * 1.25  # price increase
    df['tickets_30d'] = df['tickets_30d'] + 2              # more tickets
    df.to_csv('data/raw/current_batch.csv', index=False)


    report = detect_drift('data/raw/customers.csv', 'data/raw/current_batch.csv')
    print(f'\nDrift report saved to models/drift_report.json')
