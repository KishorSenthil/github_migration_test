import argparse
import os
import pickle
from logging import Logger

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from housing.logger import configure_logger

model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]


def get_path():
    """
    Get the absolute path to the 'mle_training' directory from the current working directory.

    Returns:
    str: The absolute path to the 'mle_training' directory.

    Note:
    This function retrieves the path by traversing upwards from the current directory until it finds
    the 'mle_training' directory. Returns the absolute path to 'mle_training' or its parent directory.

    Example:
    >>> get_path()
    '/absolute/path/to/mle_training/'
    """
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def scoring(X_test, y_test, lin_reg, tree_reg, forest_reg, grid_search):
    """
    Calculate various evaluation metrics for different regression models on test data.

    Parameters:
    X_test (array-like): Test features.
    y_test (array-like): True labels for the test data.
    lin_reg: Trained Linear Regression model.
    tree_reg: Trained Decision Tree Regression model.
    forest_reg: Trained RandomForest Regression model.
    grid_search: Trained GridSearchCV object for hyperparameter tuning.

    Returns:
    tuple: Contains lists of evaluation scores for each model in the order [lin_scores, tree_scores, forest_scores, grid_search_scores].
           Each score list contains [Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)].

    Note:
    This function computes MAE, MSE, and RMSE for predictions made by each provided model (Linear Regression, Decision Tree, RandomForest,
    and the GridSearchCV-tuned model) on the given test data.

    Example:
    >>> # Assuming X_test, y_test, lin_reg, tree_reg, forest_reg, and grid_search are trained models and test data
    >>> scoring(X_test, y_test, lin_reg, tree_reg, forest_reg, grid_search)
    >>> # Returns evaluation scores for each model in the form of nested lists
    """
    lin_predictions = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_test, lin_predictions)

    tree_predictions = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y_test, tree_predictions)

    forest_predictions = forest_reg.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(y_test, forest_predictions)

    grid_search_predictions = grid_search.predict(X_test)
    grid_search_mse = mean_squared_error(y_test, grid_search_predictions)
    grid_search_rmse = np.sqrt(grid_search_mse)
    grid_search_mae = mean_absolute_error(y_test, grid_search_predictions)

    lin_scores = [lin_mae, lin_mse, lin_rmse]
    tree_scores = [tree_mae, tree_mse, tree_rmse]
    forest_scores = [forest_mae, forest_mse, forest_rmse]
    grid_search_scores = [grid_search_mae, grid_search_mse, grid_search_rmse]

    return lin_scores, tree_scores, forest_scores, grid_search_scores


def load_data(in_path):
    """
    Load test data from specified path and return prepared features and labels.

    Parameters:
    in_path (str): The directory path containing test data.

    Returns:
    tuple: A tuple containing two elements:
            - prepared: DataFrame containing prepared features loaded from 'test_X.csv'.
            - labels: Array of labels loaded from 'test_y.csv'.

    Note:
    This function reads the test data from 'test_X.csv' and 'test_y.csv' files within the specified directory path.
    'test_X.csv' contains prepared features, and 'test_y.csv' contains corresponding labels.
    The labels are extracted and returned as a flattened array.

    Example:
    >>> # Assuming 'in_path' points to the directory containing 'test_X.csv' and 'test_y.csv'
    >>> load_data('/path/to/test_data')
    >>> # Returns prepared features DataFrame and labels array
    """
    prepared = pd.read_csv(in_path + "/test_X.csv")
    lables = pd.read_csv(in_path + "/test_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


def load_models(model_path):
    """
    Load specified models from the given directory path.

    Parameters:
    model_path (str): The directory path where models are stored.
    model_names (List[str]): List of model filenames to load (without extension).

    Returns:
    List: A list containing the loaded models.

    Note:
    This function loads multiple models from the specified 'model_path' directory.
    'model_names' is a list of filenames (without extension) representing the models to be loaded.
    The function loads each model using pickle and appends them to a list, returning the list of loaded models.

    Example:
    >>> # Assuming 'model_path' points to the directory containing the models and 'model_names' lists model filenames
    >>> load_models('/path/to/models', ['model1', 'model2', 'model3'])
    >>> # Returns a list containing loaded models
    """
    models = []
    for i in model_names:
        with open(model_path + "/models/" + i + ".pkl", "rb") as f:
            models.append(pickle.load(f))
    return models


def score(models, X_test, y_test):
    """
    Calculate scores for multiple models using test data.

    Parameters:
    models (List): A list containing loaded regression models.
    X_test: Test features.
    y_test: Test labels.

    Returns:
    List[List[float]]: A list containing scores for each model.

    Note:
    This function computes scores for each model in the 'models' list using the provided test data.
    It internally calls the 'scoring' function, passing each model and the test data, and returns a list
    containing scores for linear regression, decision tree regression, random forest regression, and grid search regression.

    Example:
    >>> # Assuming 'models' contains loaded regression models and 'X_test', 'y_test' represent test data
    >>> score(models, X_test, y_test)
    >>> # Returns a list containing scores for each model
    """
    lin_scores, tree_scores, forest_scores, grid_search_scores = scoring(
        X_test, y_test, models[0], models[1], models[2], models[3]
    )

    return [lin_scores, tree_scores, forest_scores, grid_search_scores]
