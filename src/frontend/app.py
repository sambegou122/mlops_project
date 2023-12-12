import requests
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
PREDICT_URL = os.getenv("PREDICT_URL")


st.header("Sentiment Analysis")
prediction, history = st.tabs(["Predict", "History"])
# Text area for the user to enter their text

# Button to launch the prediction
with prediction:
    text = st.text_area("Enter your text here")
    texts = text.split(",")
    if st.button("Predict"):
        # Call the predict service
        response = requests.post(f"{PREDICT_URL}/predict", json={"reviews": texts})
        
        # Check if the request was successful
        if response.status_code == 200:
            # Get the prediction from the response
            prediction = response.json()["sentiments"]
            result = (", ").join(prediction)
            # Display the result
            st.success(f"The sentiment is : {result}")
        else:
            st.error("An error occurred during the prediction.")


with history:
    # Call the history service
    if st.button("Get History"):
        response = requests.get(f"{PREDICT_URL}/history?n=10")
        # Check if the request was successful
        if response.status_code == 200:
            # Get the history from the response
            history = response.json()["history"]
            # Display the result
            st.dataframe(history)
        else:
            st.error("An error occurred during the prediction.")