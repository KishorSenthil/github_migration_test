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
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def train(housing_prepared, housing_labels):
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
    prepared = pd.read_csv(in_path + "/train_X.csv")
    lables = pd.read_csv(in_path + "/train_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


# artifacts folder - to store models.
# below code has the path to store those models.
def rem_artifacts(out_path):
    if os.path.exists(out_path + "/models"):
        shutil.rmtree(out_path + "/models")


# pikle (.pkl)file can convert complex Python objects, including machine learning models, into a byte stream.
# This allows you to save the model's state and structure in a file.
def model(lin_reg, tree_reg, forest_reg, grid_search, out_path):
    out_path = out_path + "/models"
    os.makedirs(out_path)
    pickle.dump(lin_reg, open(out_path + "/lin_model.pkl", "wb"))
    pickle.dump(tree_reg, open(out_path + "/tree_model.pkl", "wb"))
    pickle.dump(forest_reg, open(out_path + "/forest_model.pkl", "wb"))
    pickle.dump(grid_search, open(out_path + "/grid_search_model.pkl", "wb"))
