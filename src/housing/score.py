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
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def scoring(X_test, y_test, lin_reg, tree_reg, forest_reg, grid_search):
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
    prepared = pd.read_csv(in_path + "/test_X.csv")
    lables = pd.read_csv(in_path + "/test_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


def load_models(model_path):
    models = []
    for i in model_names:
        with open(model_path + "/models/" + i + ".pkl", "rb") as f:
            models.append(pickle.load(f))
    return models


def score(models, X_test, y_test):
    lin_scores, tree_scores, forest_scores, grid_search_scores = scoring(
        X_test, y_test, models[0], models[1], models[2], models[3]
    )

    return [lin_scores, tree_scores, forest_scores, grid_search_scores]
