import os
import mlflow
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
# os.environ["TEST_BASELINE_VERSION"] = "1"
# os.environ["TEST_BASELINE_MODEL"] ="logistic-regression"
# os.environ["TEST_MODEL_VERSION"] = "1"
# os.environ["TEST_MODEL_NAME"] = "hyper-opt-logistique"
# os.environ["TEST_TEST_SET"] = "./notebook/data/test.csv"

load_dotenv()

TEST_BASELINE_VERSION = os.getenv("TEST_BASELINE_VERSION")
TEST_BASELINE_MODEL = os.getenv("TEST_BASELINE_MODEL")
TEST_MODEL_VERSION = os.getenv("TEST_MODEL_VERSION")
TEST_MODEL_NAME = os.getenv("TEST_MODEL_NAME")
TEST_TEST_SET = os.getenv("TEST_TEST_SET")



mlflow.set_tracking_uri('http://127.0.0.1:5000/')


def load_model():
    """Load the model."""
    
    loaded_model = mlflow.sklearn.load_model(f"models:/{TEST_MODEL_NAME}/{TEST_MODEL_VERSION}")

    return loaded_model

def load_baseline_model():
    """Load the baseline model."""
    
    loaded_model = mlflow.sklearn.load_model(f"models:/{TEST_BASELINE_MODEL}/{TEST_BASELINE_VERSION}")

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


def test_accuracy_model():
    """Test the model accuracy on the test set."""

    model = load_model()

    test = pd.read_csv(TEST_TEST_SET, delimiter=',')
    test.drop(['Unnamed: 0', "film-url"], axis=1, inplace=True)

    assert list(test.columns) == ['review', "polarity"], \
        f"Expected columns: ['review', polarity], but got {test.columns}"
    

    predictions = model.predict(test['review'])
    accuracy = accuracy_score(test['polarity'], predictions)

    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy}"


def test_accuracy_baseline_model():
    """Test the baseline model accuracy on the test set."""

    base = load_baseline_model()
    model = load_model()

    test = pd.read_csv(TEST_TEST_SET, delimiter=',')
    test.drop(['Unnamed: 0', "film-url"], axis=1, inplace=True)

    assert list(test.columns) == ['review', "polarity"], \
        f"Expected columns: ['review', polarity], but got {test.columns}"
    

    predictions = model.predict(test['review'])
    base_predictions = base.predict(test['review'])
    accuracy = accuracy_score(test['polarity'], predictions)
    base_accuracy = accuracy_score(test['polarity'], base_predictions)

    assert accuracy > base_accuracy, f"Expected accuracy > {base_accuracy}, but got {accuracy}"



