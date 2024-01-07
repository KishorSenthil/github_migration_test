import argparse
import os
import tarfile
from six.moves import urllib
from logging import Logger

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from urllib import request as urllib

# from housing import run_script as args
from housing.logger import configure_logger

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
imputer = SimpleImputer(strategy="median")


# args = initiator.parse_args()


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download and extract the housing dataset from a given URL.

    Parameters
    ----------
    housing_url : str, optional
        The URL from which to download the housing dataset.
    housing_path : str, optional
        The local directory where the dataset will be stored.

    Returns
    -------
    None
        The function doesn't return anything. It downloads and extracts the dataset.

    Examples
    --------
    >>> fetch_housing_data()
    # Downloads and extracts the housing dataset to the default location.

    >>> fetch_housing_data(housing_url='http://example.com/housing.tgz', housing_path='/path/to/custom/location')
    # Downloads and extracts the housing dataset from a custom URL to a custom location.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load the housing dataset from a CSV file.

    Parameters
    ----------
    housing_path : str, optional
        The local directory where the dataset CSV file is located.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded housing dataset.

    Examples
    --------
    >>> load_housing_data()
    # Loads the housing dataset from the default location.

    >>> load_housing_data(housing_path='/path/to/custom/location')
    # Loads the housing dataset from a custom location.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """
    Calculate the proportions of each income category in the given dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the "income_cat" column.

    Returns
    -------
    pandas.Series
        A Series containing the proportions of each income category.

    Examples
    --------
    >>> income_cat_proportions(my_dataset)
    # Calculates the proportions of each income category in the provided dataset.
    """
    return data["income_cat"].value_counts() / len(data)


def train_test(housing):
    """
    Split the housing dataset into training and testing sets with stratified sampling based on income categories.

    Parameters
    ----------
    housing : pandas.DataFrame
        The dataset to be split.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing the stratified training and testing sets.

    Examples
    --------
    >>> train_set, test_set = train_test(my_housing_dataset)
    # Splits the housing dataset into training and testing sets using stratified sampling.
    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def preprocess(strat_train_set):
    """
    Preprocess the stratified training set for housing data.

    Parameters
    ----------
    strat_train_set : pandas.DataFrame
        The stratified training set of the housing data.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing the preprocessed housing features and labels.

    Examples
    --------
    >>> features, labels = preprocess(my_stratified_train_set)
    # Preprocesses the stratified training set of the housing data.
    """
    housing = strat_train_set.copy()

    corr_matrix = housing.corr(numeric_only=True)
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )
    return housing_prepared, housing_labels


#  to get the project's directory
def get_path():
    """
    Retrieve the path of the 'mle-training' project directory.

    Returns
    -------
    str
        The absolute path of the 'mle-training' project directory.

    Examples
    --------
    >>> project_path = get_path()
    # Retrieves the absolute path of the 'mle-training' project directory.
    """
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle_training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


# to store the trained/validated datasets
def save_preprocessed(train_X, train_y, test_X, test_y, processed):
    """
    Save preprocessed data to CSV files.

    Parameters
    ----------
    train_X : pandas.DataFrame
        Features of the training set.
    train_y : pandas.Series
        Labels of the training set.
    test_X : pandas.DataFrame
        Features of the test set.
    test_y : pandas.Series
        Labels of the test set.
    processed : str
        Path to the directory where the preprocessed files will be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> save_preprocessed(train_X, train_y, test_X, test_y, '/path/to/processed')
    # Saves preprocessed data to CSV files in the specified directory.
    """
    train_X.to_csv(os.path.join(processed, "train_X.csv"), index=False)
    train_y.to_csv(os.path.join(processed, "train_y.csv"), index=False)
    test_X.to_csv(os.path.join(processed, "test_X.csv"), index=False)
    test_y.to_csv(os.path.join(processed, "test_y.csv"), index=False)
