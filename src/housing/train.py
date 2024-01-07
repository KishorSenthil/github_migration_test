import argparse
import os
import pickle
import shutil
from logging import Logger

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from housing.logger import configure_logger

model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]


def get_path():
    """
    Get the path of the 'mle_training' directory by traversing upwards from the current working directory.

    Returns:
        str: The absolute path to the 'mle_training' directory. If not found, returns the current working directory.

    Example:
        If the current working directory is '/home/user/projects/mle_training/src/',
        get_path() will return '/home/user/projects/mle_training/'.
    """
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def train(housing_prepared, housing_labels):
    """
    Train various regression models using the provided prepared data.

    Trains a Linear Regression model, a Decision Tree Regression model, a RandomForest Regression model,
    and performs GridSearchCV to fine-tune hyperparameters for the RandomForest model.

    Parameters:
    housing_prepared (array-like): The prepared features for training.
    housing_labels (array-like): The target labels for training.

    Returns:
    tuple: A tuple containing the trained models - (lin_reg, tree_reg, forest_reg, grid_search).

    Note:
    This function trains multiple models and performs hyperparameter tuning through GridSearchCV.

    Example:
    >>> # Assuming housing_prepared and housing_labels are prepared training data
    >>> models = train(housing_prepared, housing_labels)
    >>> lin_reg, tree_reg, forest_reg, grid_search = models
    >>> # Use the trained models for predictions or further analysis
    """
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    return lin_reg, tree_reg, forest_reg, grid_search


def load_data(in_path):
    """
    Load prepared features and labels from CSV files in the specified input path.

    Parameters:
    in_path (str): The path where 'train_X.csv' and 'train_y.csv' files are located.

    Returns:
    tuple: A tuple containing the loaded prepared features and labels.

    Note:
    This function loads 'train_X.csv' as prepared features and 'train_y.csv' as labels.
    Assumes 'train_y.csv' contains a single column of target labels.

    Example:
    >>> # Assuming in_path is the path to the directory containing 'train_X.csv' and 'train_y.csv'
    >>> prepared_features, labels = load_data(in_path)
    >>> # Use the loaded data for training or analysis
    """
    prepared = pd.read_csv(in_path + "/train_X.csv")
    lables = pd.read_csv(in_path + "/train_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


# artifacts folder - to store models.
# below code has the path to store those models.
def rem_artifacts(out_path):
    """
    Remove the 'models' directory from the specified output path if it exists.

    Parameters:
    out_path (str): The path where the 'models' directory is located.

    Note:
    This function checks if the 'models' directory exists within the specified path.
    If it exists, the entire directory and its contents will be removed.

    Example:
    >>> # Assuming out_path is the path containing the 'models' directory
    >>> rem_artifacts(out_path)
    >>> # 'models' directory and its contents will be deleted if present
    """
    if os.path.exists(out_path + "/models"):
        shutil.rmtree(out_path + "/models")


# pikle (.pkl)file can convert complex Python objects, including machine learning models, into a byte stream.
def model(lin_reg, tree_reg, forest_reg, grid_search, out_path):
    """
    Save trained regression models and GridSearchCV object to pickle files within a 'models' directory.

    Parameters:
    lin_reg: Trained Linear Regression model.
    tree_reg: Trained Decision Tree Regression model.
    forest_reg: Trained RandomForest Regression model.
    grid_search: Trained GridSearchCV object for hyperparameter tuning.
    out_path (str): The path where the 'models' directory will be created to store the models.

    Note:
    This function saves the provided regression models and GridSearchCV object as pickle files.
    Creates a 'models' directory within the specified output path to store these files.

    Example:
    >>> # Assuming lin_reg, tree_reg, forest_reg, and grid_search are trained models
    >>> model(lin_reg, tree_reg, forest_reg, grid_search, "/path/to/output")
    >>> # Trained models will be saved as pickle files within the 'models' directory
    """
    out_path = out_path + "/models"
    os.makedirs(out_path)
    pickle.dump(lin_reg, open(out_path + "/lin_model.pkl", "wb"))
    pickle.dump(tree_reg, open(out_path + "/tree_model.pkl", "wb"))
    pickle.dump(forest_reg, open(out_path + "/forest_model.pkl", "wb"))
    pickle.dump(grid_search, open(out_path + "/grid_search_model.pkl", "wb"))
