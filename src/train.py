import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import yaml
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns




def load_params(path='params.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)




def load_splits(splits_dir):
    X_train = pd.read_csv(f'{splits_dir}/X_train.csv')
    X_test  = pd.read_csv(f'{splits_dir}/X_test.csv')
    y_train = pd.read_csv(f'{splits_dir}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{splits_dir}/y_test.csv').squeeze()
    preprocessor = joblib.load(f'{splits_dir}/preprocessor.pkl')
    return X_train, X_test, y_train, y_test, preprocessor




def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)




def train(params_path='params.yaml', splits_dir='data/splits'):
    params = load_params(params_path)
    X_train, X_test, y_train, y_test, preprocessor = load_splits(splits_dir)


    mlflow.set_experiment('churn_prediction')


    with mlflow.start_run(run_name='random_forest_v1') as run:
        # Log parameters
        model_params = params['model']
        mlflow.log_params(model_params)
        mlflow.log_param('dataset_version', 'v1.0')
        mlflow.log_param('n_train', len(X_train))
        mlflow.log_param('n_test', len(X_test))
        

        # Build full pipeline (preprocessor + classifier)
        clf = RandomForestClassifier(**model_params)
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])


        # Train
        model_pipeline.fit(X_train, y_train)


        # Evaluate
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]


        f1  = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)


        # Log metrics
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', roc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)


        print(f'F1 Score:  {f1:.4f}')
        print(f'ROC-AUC:   {roc:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall:    {rec:.4f}')
        print(classification_report(y_test, y_pred))

        metrics_dict = {'f1_score': f1, 'roc_auc': roc, 'precision': prec, 'recall': rec}
        # Log confusion matrix artifact
        os.makedirs('models', exist_ok=True)
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)    
        plot_confusion_matrix(y_test, y_pred, 'models/confusion_matrix.png')
        mlflow.log_artifact('models/confusion_matrix.png')


        # Log and register model
        mlflow.sklearn.log_model(
            model_pipeline,
            artifact_path='model',
            registered_model_name='ChurnPredictionModel'
        )


        # Save locally
        joblib.dump(model_pipeline, 'models/churn_model.pkl')
        print(f'Model saved. Run ID: {run.info.run_id}')


        return model_pipeline, run.info.run_id




if __name__ == '__main__':
    train()
