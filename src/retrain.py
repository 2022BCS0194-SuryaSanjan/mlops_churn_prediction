import json
import subprocess
import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient




def check_drift_and_retrain(drift_report_path='models/drift_report.json',
                             auto_promote=False):
    """Read drift report and retrain if drift is detected."""
    if not os.path.exists(drift_report_path):
        print('No drift report found. Run monitor.py first.')
        return


    with open(drift_report_path) as f:
        report = json.load(f)


    print(f'Drift status: {report["overall_status"]}')
    print(f'Report timestamp: {report["timestamp"]}')


    if report['overall_status'] == 'DRIFT_DETECTED':
        print('\nDrift detected. Starting retraining pipeline...')


        # Run DVC pipeline to reproduce with new data
        result = subprocess.run(['dvc', 'repro', '--force'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print('ERROR during retraining:', result.stderr)
            return


        print('Retraining complete!')


        # Auto-promote if specified
        if auto_promote:
            client = MlflowClient()
            experiment = mlflow.get_experiment_by_name('churn_prediction')
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['start_time DESC'],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                print(f'Latest run: {run_id}')
    else:
        print('No retraining needed.')




if __name__ == '__main__':
    check_drift_and_retrain(auto_promote=False)
