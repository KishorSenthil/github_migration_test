import argparse
import os
import warnings
import mlflow
import mlflow.sklearn

from housing import ingest_data as data
from housing import score as scoreses
from housing import train as trains

# Filter out DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ingest_data.py


# we add argparse here. The option '--datapath' will accept path as an argument from the user, which will be used to store the training and validation datasets.
# The script that accepts the output folder/file path as a user argument.
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
        "--outputpath",
        help="path to store the output ",
        type=str,
        default="artifacts",
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
        "--log-path", type=str, default=data.get_path() + "logs/logs.log"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="housing_experiment"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # this for ingest_data.py
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    # Main run
    with mlflow.start_run():
        logger = data.configure_logger(
            log_level=args.log_level,
            log_file=args.log_path,
            console=not args.no_console_log,
        )

        parent_path = data.get_path()
        path = parent_path + args.datapath
        mlflow.log_param("datapath", args.datapath)
        mlflow.log_param("dataprocessed", args.dataprocessed)
        mlflow.log_param("inputpath", args.inputpath)
        mlflow.log_param("outputpath", args.outputpath)
        mlflow.log_param("modelpath", args.modelpath)
        mlflow.log_param("log-level", args.log_level)
        mlflow.log_param("no-console-log", args.no_console_log)
        mlflow.log_param("log-path", args.log_path)

        # Child run 1: data preparation
        with mlflow.start_run(nested=True):
            mlflow.log_param("run_type", "data_preparation")
            data.fetch_housing_data(housing_path=path)
            logger.debug("Fetched housing data.")
            logger.debug(f"Dataset stored at {path}.")
            housing_csv = data.load_housing_data(housing_path=path)
            logger.debug("Loaded housing data.")
            train, test = data.train_test(housing_csv)
            train_X, train_y = data.preprocess(train)
            mlflow.log_param("train_X_shape", train_X.shape)
            mlflow.log_param("train_y_shape", train_y.shape)
            logger.debug("Preprocessing housing data...")
            test_X, test_y = data.preprocess(test)
            processed = parent_path + args.dataprocessed
            if not os.path.exists(processed):
                os.makedirs(processed)
            data.save_preprocessed(train_X, train_y, test_X, test_y, processed)
            logger.debug(
                f"Preprocessed train and test datasets stored at {processed}."
            )
            mlflow.log_artifact(
                processed
            )  # Log the preprocessed data as an artifact

        # Child run 2: model training
        with mlflow.start_run(nested=True):
            mlflow.log_param("run_type", "model_training")
            path_parent = trains.get_path()
            in_path = path_parent + args.inputpath
            out_path = path_parent + args.outputpath
            mlflow.log_param("inputpath", args.inputpath)
            mlflow.log_param("outputpath", args.outputpath)
            mlflow.log_param("log-level", args.log_level)

            trains.rem_artifacts(out_path)
            prepared, labels = trains.load_data(in_path)
            logger.debug("Loaded training data")
            lin_reg, tree_reg, forest_reg, grid_search = trains.train(
                prepared, labels
            )
            logger.debug("Training completed")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            trains.model(lin_reg, tree_reg, forest_reg, grid_search, out_path)
            mlflow.log_param("linear ", lin_reg)
            mlflow.log_param("tree reg", tree_reg)
            mlflow.log_param("forest reg", forest_reg)
            mlflow.log_param("grid_search", grid_search)
            mlflow.log_artifact(
                out_path
            )  # Log the trained models as an artifact

        # Child run 3: scoring
        with mlflow.start_run(nested=True):
            mlflow.log_param("run_type", "scoring")
            path_parent = scoreses.get_path()
            data_path = path_parent + args.dataprocessed
            model_path = path_parent + args.modelpath
            X_test, y_test = scoreses.load_data(data_path)
            logger.debug("Loaded test data")
            models = scoreses.load_models(model_path)
            logger.debug("Loaded Models")
            scores = scoreses.score(models, X_test, y_test)
            for i in range(len(models)):
                logger.debug(f"{scoreses.model_names[i]}={scores[i]}")
                mlflow.log_metric(
                    f"{scoreses.model_names[i]}_MAE", scores[i][0]
                )
                mlflow.log_metric(
                    f"{scoreses.model_names[i]}_MSE", scores[i][1]
                )
                mlflow.log_metric(
                    f"{scoreses.model_names[i]}_RMSE", scores[i][2]
                )
