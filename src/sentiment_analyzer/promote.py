import sys
import mlflow
import click
import os
import subprocess
import pkg_resources
from dotenv import load_dotenv 
import pandas as pd
load_dotenv()

@click.command()
@click.option('--model_name', type=str, required=True, help='Model name in MLFlow registry')
@click.option('--model_version', type=str, required=True, help='Model version in MLFlow registry')
@click.option('--status', type=str, required=True, help='Model status to promote to')
@click.option('--test_set', default = "./notebook/data/test.csv", type=click.Path(exists=True), help='Test set path')





def promote(model_name, model_version, status, test_set):
    """Promote the model to the next stage."""
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
   
    # model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    client = mlflow.tracking.MlflowClient()
    model = client.get_model_version(name=model_name, version=model_version)

    print(f"Promoting model {model_name}/{model_version} to {status}")

    os.environ["TEST_TEST_SET"]  = test_set
    os.environ["TEST_MODEL_VERSION"] = model_version
    os.environ["TEST_MODEL_NAME"] = model_name

    # Check if the model can be promoted to the next stage
    if status not in ['Staging', 'Production', 'Archived'] or \
       (model.current_stage == 'None' and status != 'Staging') or \
       (model.current_stage == 'Staging' and status != 'Production') or \
       (model.current_stage == 'Production' and status != 'Archived'):
        print(f"Cannot promote model from {model.current_stage} to {status}")
        sys.exit(1)
    

    # If promoting to Production, run the tests
    if status == 'Staging':
        print("Running tests...")
        test_result = subprocess.run(
            ["pytest",pkg_resources.resource_filename('sentiment_analyzer', "./tests")], \
            capture_output=False)
        if test_result.returncode != 0:
            print("Tests failed. Model not promoted")
            sys.exit(1)
        else:
            print("Tests passed. Model promoted to Staging")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging",
            )

    if status == 'Production':
        print("Running tests...")
        test_result = subprocess.run(
            ["pytest",pkg_resources.resource_filename('sentiment_analyzer', "./tests")], \
            capture_output=False)
        if test_result.returncode != 0:
            print("Tests failed. Model not promoted")
            sys.exit(1)
        else:
            print("Tests passed. Model promoted to Production")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production",
            )
    
    if status == 'Archived':
        print("Running tests...")
        test_result = subprocess.run(
            ["pytest",pkg_resources.resource_filename('sentiment_analyzer', "./tests")], \
            capture_output=False)
        if test_result.returncode != 0:
            print("Tests failed. Model not promoted")
            sys.exit(1)
        else:
            print("Tests passed. Model promoted to Archived")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Archived",
            )



    



if __name__ == '__main__':
    promote()