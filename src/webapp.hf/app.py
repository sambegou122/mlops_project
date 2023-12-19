from sklearn.externals import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from pymongo import MongoClient


model = joblib.load("/model/model.pkl")

## Connect to MongoDB
client  = MongoClient("mongodb://mongo:27017/")
db = client["sentiment_analyzer"]
collection = db["history"]


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
    try:
        logger.info("Received request for prediction")
        # Your prediction code here

        predictions = model.predict(input.reviews).tolist()
        predictions = [ "positive" if x == 1 else "negative" for x in predictions]
        collection.insert_one({"reviews": input.reviews, "sentiments": predictions})

        logger.debug(f"Input: {input.reviews}, Output: {predictions}")
        return {"sentiments": predictions}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e
    
@app.get("/history", summary="Get prediction history")
def history(n :int):

    """
    Get prediction history

    **return**: list of predictions

    """
    try:
        logger.info("Received request for history")
        # Your prediction code here
        history = list(collection.find().sort("_id", -1).limit(n))

        logger.debug(f"Output: {history}")
        return {"history": history}
    except Exception as e:
        logger.error(f"Error during history: {e}")
        raise e
    

    






