# Start from a Python 3.9 image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /frontend

# Install Streamlit and requests
RUN pip install streamlit requests python-dotenv loguru

# Copy the Streamlit app into the container
COPY ./app.py .

ENV PREDICT_URL=$PREDICT_URL

# Expose the Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py"]