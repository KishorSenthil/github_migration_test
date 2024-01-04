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
    Retrieves the absolute path of the 'mle_training' directory or the closest parent directory
    named 'mle_training' starting from the current working directory.

    Returns:
    str: Absolute path of the 'mle_training' directory with a trailing slash.
         If the directory is not found, returns the current working directory path.

    Raises:
    No specific exceptions are raised within this function.
    """
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def parse_args():
     """
    Parses command-line arguments using argparse.

    Returns:
    argparse.Namespace: An object containing attributes corresponding to the command-line arguments.

    The function sets up an argument parser with the following options:
    1. --datapath: Path to the datasets. Default is 'data/processed'.
    2. --modelpath: Path to the model files. Default is 'artifacts'.
    3. --log-level: Logging level. Default is 'DEBUG'.
    4. --no-console-log: Flag to suppress console logging.
    5. --log-path: Path to the log file. Default is 'logs/logs.log' relative to 'mle_training' directory.

    Note:
    'get_path()' is a function used to retrieve the absolute path of the 'mle_training' directory
    or the closest parent directory named 'mle_training' from the current working directory.
    If 'get_path()' fails to find the directory, it will default to the current working directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="path to the datasets ",
        type=str,
        default="data/processed",
    )
    parser.add_argument(
        "--modelpath",
        help="path to the model files ",
        type=str,
        default="artifacts",
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument(
        "--log-path", type=str, default=get_path() + "logs/logs.log"
    )
    return parser.parse_args()


def scoring(X_test, y_test, reg):
    predictions = reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    scores = [mae, mse, rmse]

    return scores


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
    result = []
    for i in range(len(models)):
        result.append(scoring(X_test, y_test, models[i]))

    return result


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    path_parent = get_path()
    data_path = path_parent + args.datapath
    model_path = path_parent + args.modelpath
    X_test, y_test = load_data(data_path)
    logger.debug("Loaded test data")
    models = load_models(model_path)
    logger.debug("Loaded Models")
    scores = []
    scores = score(models, X_test, y_test)
    for i in range(len(models)):
        logger.debug(f"{model_names[i]}={scores[i]}")
