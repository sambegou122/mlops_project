import click
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
import os
load_dotenv()


model_name = os.getenv("TEST_BASELINE_MODEL")
model_version = os.getenv("TEST_BASELINE_VERSION")
trarget_path = os.getenv("SENTIMENT_ANALYZER_MODEL_PATH")
mlflow_server_uri = os.getenv("MLFLOW_SERVER_URI")

@click.command()
@click.option("--mlflow_server_uri", default = mlflow_server_uri)
@click.option("--mlflow_model_name", default = model_name)
@click.option("--mlflow_model_version", default = model_version)
@click.option("--target_path", default = trarget_path)

def main(mlflow_server_uri, mlflow_model_name, mlflow_model_version, target_path):
    mlflow.set_tracking_uri(mlflow_server_uri)
    # client = mlflow.tracking.MlflowClient()
    # model = client.get_model_version(name=mlflow_model_name, version=mlflow_model_version)
    # mlflow.sklearn.save_model(model, target_path)
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{mlflow_model_name}/{mlflow_model_version}",
    )
    mlflow.sklearn.save_model(model, target_path)




if __name__ == "__main__":
    main()
    