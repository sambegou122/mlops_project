import click
import mlflow
from sentiment_analyzer.model_manager import ModelManager

@click.command()
@click.option('--input_file', type=click.Path(exists=True), help='Input file path')
@click.option('--output_file', type=click.Path(), help='Output file path')
@click.option('--text', type=str, help='Text to predict')
@click.option('--model_name', type=str, required=True, help='Model name in MLFlow registry')
@click.option('--model_version', type=str, required=True, help='Model version in MLFlow registry')
@click.option('--mlflow_url', default='http://127.0.0.1:5000/', help='MLFlow server URL')



def predict(input_file, output_file, text, model_name, model_version, mlflow_url):
    if not input_file and not text:
        raise click.UsageError('You must provide either --input_file or --text')
    if input_file and text:
        raise click.UsageError('You can only provide either --input_file or --text, not both')
    
    mlflow.set_tracking_uri(mlflow_url)
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    model_manager = ModelManager(model, text, input_file, output_file)
    prediction = model_manager.predict()
    print(f"Prediction: {prediction}")



if __name__ == '__main__':
    predict()