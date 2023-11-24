import click
import mlflow
import pandas as pd
from sentiment_analyzer.model_manager import ModelManager
from dotenv import load_dotenv
import os
load_dotenv()

model_name = os.getenv("TEST_BASELINE_MODEL")
model_version = os.getenv("TEST_BASELINE_VERSION")
training_set = "./notebook/data/train.csv"

@click.command()
@click.option('--model_name', type=str, default = model_name, help='Model name in MLFlow registry')
@click.option('--model_version', type=str, default = model_version, help='Model version in MLFlow registry')
@click.option('--training_set', type=click.Path(exists=True), default = training_set, help='Training set path')
@click.option('--training_set_id', type=str, default='default_id', help='Training set ID')
@click.option('--register_updated_model', is_flag=True, help='Register the updated model')
@click.option('--mlflow_url', default='http://127.0.0.1:5000/', help='MLFlow server URL')


def retrain(self, model_name, model_version, training_set, \
            training_set_id, register_updated_model,mlflow_url):
    
    """Retrain the model."""

    # Load the model
    mlflow.set_tracking_uri(mlflow_url)
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Load the training data
    
    model_manager = ModelManager(model, None, None, None)
    model_manager.retrain(training_set)


    # Set the tags
    tags = {
        'retrained': 'True',
        'parent_version': model_version,
    }
    if training_set_id!='default_id':
        tags['training_set_id'] = training_set_id

    # Log the retrained model
    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.set_tags(tags)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name=model_name
        )
    
        # Register the updated model if requested
        if register_updated_model:
            client = mlflow.MlflowClient()
            client.create_registered_model(model_name)
            client.create_model_version(model_name, run.info.run_id, run.info.artifact_uri)



if __name__ == '__main__':
    retrain()