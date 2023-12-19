# Start from a lightweight generic image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /webapp

# Install required packages
RUN pip install mlflow fastapi uvicorn python-dotenv loguru pymongo

# Copy the required source files
COPY app.py .
COPY get_mlflow_model.py .

# Arguments for the model retrieval script
ARG MLFLOW_SERVER_URI MODEL_NAME MODEL_VERSION

# Run the model retrieval script
RUN python get_mlflow_model.py --mlflow_server_uri $MLFLOW_SERVER_URI --mlflow_model_name $MODEL_NAME --mlflow_model_version $MODEL_VERSION --target_path /model

# Install the model requirements
RUN pip install -r /model/requirements.txt

# Expose port 8000
EXPOSE 8000
ENV SENTIMENT_ANALYZER_MODEL_PATH=/model

# Start the webapp
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]