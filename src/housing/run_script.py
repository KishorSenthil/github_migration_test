import argparse
import os
import warnings

from housing import ingest_data as data
from housing import score as scoreses
from housing import train as trains

# Filter out DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ingest_data.py


# we add argparse here. The option '--datapath' will accept path as an argument from the user, which will be used to store the training and validation datasets.
# The script that accepts the output folder/file path as an user argument.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="path to store the dataset ",
        type=str,
        default="data/raw/housing",
    )
    parser.add_argument(
        "--dataprocessed",
        help="path to store the dataset ",
        type=str,
        default="data/processed",
    )
    parser.add_argument(
        "--inputpath",
        help="path to the input dataset ",
        type=str,
        default="data/processed/",
    )
    parser.add_argument(
        "--outputpath", help="path to store the output ", type=str, default="artifacts"
    )
    parser.add_argument(
        "--modelpath", help="path to the model files ", type=str, default="artifacts"
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument(
        "--log-path", type=str, default=data.get_path() + "logs/logs.log"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = data.configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    parent_path = data.get_path()
    path = parent_path + args.datapath
    data.fetch_housing_data(housing_path=path)
    logger.debug("Fetched housing data.")
    logger.debug(f"Dataset stored at {path}.")
    housing_csv = data.load_housing_data(housing_path=path)
    logger.debug("Loaded housing data.")
    train, test = data.train_test(housing_csv)
    train_X, train_y = data.preprocess(train)
    print(train_X.shape, train_y.shape)
    logger.debug("Preprocessing housing data...")
    test_X, test_y = data.preprocess(test)
    processed = parent_path + args.dataprocessed
    if not os.path.exists(processed):
        os.makedirs(processed)
    data.save_preprocessed(train_X, train_y, test_X, test_y, processed)
    logger.debug(f"Preprocessed train and test datasets stored at {processed}.")

    path_parent = trains.get_path()
    in_path = path_parent + args.inputpath
    out_path = path_parent + args.outputpath
    trains.rem_artifacts(out_path)
    prepared, labels = trains.load_data(in_path)
    logger.debug("Loaded training data")
    lin_reg, tree_reg, forest_reg, grid_search = trains.train(prepared, labels)
    logger.debug("Training completed")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    trains.model(lin_reg, tree_reg, forest_reg, grid_search, out_path)
    logger.debug(f"Models stored at {out_path}.")

    path_parent = scoreses.get_path()
    data_path = path_parent + args.dataprocessed
    model_path = path_parent + args.modelpath
    X_test, y_test = scoreses.load_data(data_path)
    logger.debug("Loaded test data")
    models = scoreses.load_models(model_path)
    logger.debug("Loaded Models")
    scores = []
    scores = scoreses.score(models, X_test, y_test)
    for i in range(len(models)):
        logger.debug(f"{scoreses.model_names[i]}={scores[i]}")
# ---------------------------------------------------------------------------------------------


# # train.py
# if __name__ == "__main__":
#     args = parse_args()
#     logger = score.configure_logger(
#         log_level=args.log_level,
#         log_file=args.log_path,
#         console=not args.no_console_log,
#     )
#     path_parent = train.get_path()
#     in_path = path_parent + args.inputpath
#     out_path = path_parent + args.outputpath
#     train.rem_artifacts(out_path)
#     prepared, labels = train.load_data(in_path)
#     logger.debug("Loaded training data")
#     lin_reg, tree_reg, forest_reg, grid_search = train(prepared, labels)
#     logger.debug("Training completed")
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     train.model(lin_reg, tree_reg, forest_reg, grid_search, out_path)
#     logger.debug(f"Models stored at {out_path}.")


# The script that accepts arguments for input (dataset) and output folders (model pickles).
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--inputpath",
#         help="path to the input dataset ",
#         type=str,
#         default="data/processed/",
#     )
#     parser.add_argument(
#         "--outputpath", help="path to store the output ", type=str, default="artifacts"
#     )
#     parser.add_argument("--log-level", type=str, default="DEBUG")
#     parser.add_argument("--no-console-log", action="store_true")
#     parser.add_argument(
#         "--log-path", type=str, default=train.get_path() + "logs/logs.log"
#     )
#     return parser.parse_args()


# ---------------------------------------------------------------------------------------------


# score.py
# if __name__ == "__main__":
#     args = parse_args()
#     logger = score.configure_logger(
#         log_level=args.log_level,
#         log_file=args.log_path,
#         console=not args.no_console_log,
#     )
#     path_parent = score.get_path()
#     data_path = path_parent + args.datapath
#     model_path = path_parent + args.modelpath
#     X_test, y_test = score.load_data(data_path)
#     logger.debug("Loaded test data")
#     models = score.load_models(model_path)
#     logger.debug("Loaded Models")
#     scores = []
#     scores = score(models, X_test, y_test)
#     for i in range(len(models)):
#         logger.debug(f"{score.model_names[i]}={scores[i]}")


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--datapath", help="path to the datasets ", type=str, default="data/processed"
#     )
#     parser.add_argument(
#         "--modelpath", help="path to the model files ", type=str, default="artifacts"
#     )
#     parser.add_argument("--log-level", type=str, default="DEBUG")
#     parser.add_argument("--no-console-log", action="store_true")
#     parser.add_argument(
#         "--log-path", type=str, default=score.get_path() + "logs/logs.log"
#     )
#     return parser.parse_args()
