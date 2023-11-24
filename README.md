# MLOps Projet 

## Project description

This project aims to set up an MLOps pipeline for a machine learning model. The model is a binary classification model that can predict whether a comment is positive or negative.

## Project structure

```
|-- README.md
|-- requirements.txt
|-- Makefile
|-- notebooks
    |-- model_design.ipynb
    |-- model_design_2.ipynb
    |-- model_design_3.ipynb
    |-- exploratory_analysis .ipynb
|-- src
    |-- sentiment_analyzer
    |   |-- __init__.py
    |   |-- model_manager.py
    |   |-- predict.py
    |   |-- retrain.py
    |   |-- promote.py
    |   |-- tests
    |       |-- __init__.py
    |       |-- test_model.py
    
```	
## Notebooks
In the notebooks folder, there are two kind of notebooks. The first notebook (model_design) allows you to design the machine learning model. The second notebook (exploratory_analysis)
contains analysis data

## Source code

The source code is in the src folder. The src folder contains the sentiment_analyzer package. This package contains the tests folder and the following files: 
- model_manager.py: This file contains the ModelManager class. This class allows you to manage the model. It allows you to retrain, predict the model
- predict.py: This file contains the predict function. This function allows you to predict with the model
- retrain.py: This file contains the retrain function. This function allows you to retrain the model
- promote.py: This file contains the promote function. This function allows you to promote the model to a new version
- tests: This folder contains the tests of the sentiment_analyzer package

## Data
The data used in this project is the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The data is divided into 160,000 reviews for training 20,000 reviews for the valid and 20,000 reviews for testing. All the dataset are balanced, meaning they contain an equal number of positive and negative reviews.

## Objective
The objective of this project is to design a machine learning model that can predict whether a comment is positive or negative.

We try the best model possible by using different techniques such as:

- Hyperparameter tuning with Hyperopt
- Using different machine learning algorithms
- Using mlflow to track the model performance and the model parameters and to save the best model


## Setting up the work environment
To be able to use the project
First, you must create a virtual environment and then install the requirements.txt file

```bash
pip install -r requirements.txt
```
Then you must install the sentiment_analyzer package

```bash
pip install -e .
```

## Running the tests
To run the tests, you must execute the make command:

```bash
make test
```

## Training the model
To train the model, you must execute the make command:

```bash
 train --model-name <model_name> --model-version <model_version> --training_set <data_path> ---training_set_id <training_set_id> --register_updated_model <register_updated_model>
```
Options:
- model-name: The name of the model to train
- model-version: The version of the model to train
- training_set: The path to the training set
- training_set_id: The id of the training set, this option is optional
- register_updated_model: If this option is set to True, the model will be registered in mlflow with new version. Else the model will be trained without registering it in mlflow

## Predicting with the model
To predict with the model, you must execute the make command:

```bash
 predict --model-name <model_name> --model-version <model_version> --input_file <data_path> --output_file <output_path> --text <text> 

```

Options:
- model-name: The name of the model to predict with
- model-version: The version of the model to predict with
- input_file: The path to the input file if you want to predict with a file
- output_file: The path to the output file if you want to predict with a file
- text: The text to predict with if you want to predict with a text
Note: You must specify either the input_file or the text option







