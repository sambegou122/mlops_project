import os
import mlflow
import pandas as pd
import os
import numpy as np
TEST_MODEL_NAME = os.environ.get("TEST_MODEL_NAME")
TEST_MODEL_VERSION = os.environ.get("TEST_MODEL_VERSION")


mlflow.set_tracking_uri('http://127.0.0.1:5000/')


def load_model():
    """Load the model."""
    
    loaded_model = mlflow.sklearn.load_model(f"models:/{TEST_MODEL_NAME}/1")

    return loaded_model



def test_sortie_model():
    """Test the model."""

    input_data = "mon commentaire n'a pas de sens"

    model = load_model()

    output = model.predict(pd.DataFrame(data = {"review": input_data}, index = [0]))[0]

    assert type(output) == np.int64

    assert output in [0,1]

def test_entreunusuelles_model():
    """Test the model."""

    input_data = "//µµ***@###"

    model = load_model()

    output = model.predict([input_data])[0]

    assert type(output) == np.int64, f"Expected {np.int64}, but got {type(output)}"

    assert output in [0,1], f"Expected {0} or {1}, but got {output}"

def test_output_model():
    """Test the model output for some obvious cases."""

    model = load_model()

    # Test case 1: obvious input 1
    input_data = "ce film est super mauvais"
    expected_output = 0  # Replace with the expected output
    output = model.predict([input_data])[0]
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test case 2: obvious input 2
    input_data = "il est genial, les acteurs sont incroyable !"
    expected_output = 1  # Replace with the expected output
    output = model.predict([input_data])[0]
    assert output == expected_output, f"Expected {expected_output}, but got {output}"







