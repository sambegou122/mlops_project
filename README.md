# MLOps Projet 

## Project description

This project aims to set up an MLOps pipeline for a machine learning model. The model is a binary classification model that can predict whether a comment is positive or negative.

## Project structure

```
|-- README.md
|-- requirements.txt
|-- notebooks
    |-- model_design.ipynb
    |-- model_design_2.ipynb
    |-- model_design_3.ipynb
    |-- exploratory_analysis .ipynb
```	
## Notebooks
In the notebooks folder, there are two kind of notebooks. The first notebook (model_design) allows you to design the machine learning model. The second notebook (exploratory_analysis)
contains analysis data

## Data
The data used in this project is the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The data is divided into 160,000 reviews for training 20,000 reviews for the valid and 20,000 reviews for testing. All the dataset are balanced, meaning they contain an equal number of positive and negative reviews.

## Objective
The objective of this project is to design a machine learning model that can predict whether a comment is positive or negative.

We try the best model possible by using different techniques such as:

- Hyperparameter tuning with Hyperopt
- Using different machine learning algorithms
- Using mlflow to track the model performance and the model parameters and to save the best model


## Setting up the work environment
To be able to run the notebooks, you must install the project dependencies. To do this, you must execute the following command:

```bash
pip install -r requirements.txt
```




