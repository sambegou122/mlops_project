import click
import mlflow
import pandas as pd
from skops import hub_utils
import tempfile
import pickle
@click.command()
@click.option("--mlflow_url", default="http://127.0.0.1:5000", help="URL of the MLFlow server")
@click.option("--model_name", default="hyper-opt-logistique", help="Name of the model as registered in the MLFlow registry")
@click.option("--model_version", default="1", help="Version of the model as registered in the MLFlow registry")
@click.option("--data_file", default="./notebook/data/test.csv", help="The data file used to learn the model")
@click.option("--hf_id", default ="Diallo", help="Your Huggingface ID")
@click.option("--hf_token", default="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", help="Your Huggingface write access token")
def hf_export(mlflow_url, model_name, model_version, data_file, hf_id, hf_token):
    # Set MLFlow tracking URI
    data = pd.read_csv(data_file)
    mlflow.set_tracking_uri(mlflow_url)
    model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}",
        )
    # Download the model from MLFlow
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.sklearn.save_model(model, tmp_dir)
        
        f = open(tmp_dir + "/requirements.txt", 'r')
        requirements = f.readlines()
        model = tmp_dir + "/model.pkl"
        hub_utils.init(
            model = model, 
            requirements= requirements,
            dst = tmp_dir+"/tmp",
            task = 'text-classification', 
            data = data
            )
        
        hf_repo_name = f"{hf_id}/{model_name.replace('_', '-')}-{model_version}"
        hub_utils.push(model, hf_repo_name, hf_token)

if __name__ == "__main__":
    hf_export()