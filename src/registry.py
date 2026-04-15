import mlflow
from mlflow.tracking import MlflowClient


client = MlflowClient()


def promote_to_staging(model_name='ChurnPredictionModel', version=1):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage='Staging'
    )
    print(f'Model {model_name} v{version} promoted to Staging')




def promote_to_production(model_name='ChurnPredictionModel', version=1):
    # Archive any existing production models
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == 'Production':
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage='Archived'
            )
    # Promote new version
    client.transition_model_version_stage(
        name=model_name, version=version, stage='Production'
    )
    print(f'Model {model_name} v{version} promoted to Production')




def list_models(model_name='ChurnPredictionModel'):
    print(f'\nModel versions for {model_name}:')
    for mv in client.search_model_versions(f"name='{model_name}'"):
        print(f'  Version {mv.version}: {mv.current_stage}')




if __name__ == '__main__':
    promote_to_staging(version=1)
    list_models()
    promote_to_production(version=1)
    list_models()
