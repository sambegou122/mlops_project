from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import uvicorn
from dotenv import load_dotenv
import os
load_dotenv()
model_path = os.getenv("SENTIMENT_ANALYZER_MODEL_PATH")

model = mlflow.sklearn.load_model(model_path)

app=FastAPI(title="Sentiment Analyzer", version="1", destription="Sentiment Analyzer API")

class PredictInput(BaseModel):
    reviews: list[str]



@app.post("/predict", summary="Predict sentiment of reviews")
def predict(input:PredictInput):
    """
    Predict sentiment of reviews

    **param input**: list of reviews

    **return**: list of sentiments

    """
    predictions = model.predict(input.reviews).tolist()
    predictions = [ "positive" if x == 1 else "negative" for x in predictions]

    return {"sentiments": predictions}




