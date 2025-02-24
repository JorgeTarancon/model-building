"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
from time import gmtime, strftime

import boto3
import numpy as np
import pandas as pd

import mlflow
from mlflow.data.pandas_dataset import PandasDataset

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--tracking-server-arn", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--output-s3-prefix", type=str, required=False)
    args = parser.parse_args()

    input_data = args.input_data
    tracking_server_arn = args.tracking_server_arn
    experiment_name = args.experiment_name
    output_s3_prefix = args.output_s3_prefix

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        pipeline_run = mlflow.start_run(run_name=experiment_name)
        run = mlflow.start_run(run_name=f"processing-{suffix}", nested=True)

        # Load data
        logger.debug("Reading downloaded data.")
        df_data = pd.read_csv(input_data, sep=";")
        
        input_dataset = mlflow.data.from_pandas(df_data, source=input_data)
        mlflow.log_input(input_dataset, context="raw_input")
            
        target_col = "y"

        # Indicator variable to capture when pdays takes a value of 999
        df_data["no_previous_contact"] = np.where(df_data["pdays"] == 999, 1, 0)

        # Indicator for individuals not actively employed
        df_data["not_working"] = np.where(
            np.in1d(df_data["job"], ["student", "retired", "unemployed"]), 1, 0
        )

        # remove data not used for the modelling
        df_model_data = df_data.drop(
            ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
            axis=1,
        )

        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']

        df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
        df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
        df_model_data.drop('age', axis=1, inplace=True)
        df_model_data.drop('age_range', axis=1, inplace=True)

        # Scale features
        scaled_features = ['pdays', 'previous', 'campaign']
        df_model_data[scaled_features] = MinMaxScaler().fit_transform(df_model_data[scaled_features])

        # Convert categorical variables to sets of indicators
        df_model_data = pd.get_dummies(df_model_data, dtype=int)  

        # Replace "y_no" and "y_yes" with a single label column, and bring it to the front:
        df_model_data = pd.concat(
            [
                df_model_data["y_yes"].rename(target_col),
                df_model_data.drop(["y_no", "y_yes"], axis=1),
            ],
            axis=1,
        )

        model_dataset = mlflow.data.from_pandas(df_data)
        mlflow.log_input(model_dataset, context="model_dataset")

        # Shuffle and split the dataset
        train_data, validation_data, test_data = np.split(
            df_model_data.sample(frac=1, random_state=1729),
            [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
        )

        print(f"## Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")

        mlflow.log_params(
            {
                "full_dataset": df_model_data.shape,
                "train": train_data.shape,
                "validate": validation_data.shape,
                "test": test_data.shape
            }
        )

        df_train = pd.DataFrame(train_data)
        df_validation = pd.DataFrame(validation_data)
        df_test = pd.DataFrame(test_data)
        df_baseline = df_model_data.drop([target_col], axis=1)

        logger.info("Writing out datasets to %s.", base_dir)
        df_train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
        df_validation.to_csv(
            f"{base_dir}/validation/validation.csv", header=False, index=False
        )
        df_test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
        df_baseline.to_csv(f"{base_dir}/baseline/baseline.csv", header=False, index=False)
        

        logger.info("Writing out datasets to %s.", output_s3_prefix)
        train_data.to_csv(f"{output_s3_prefix}/train/train.csv", index=False, header=False)
        validation_data.to_csv(f"{output_s3_prefix}/validation/validation.csv", index=False, header=False)
        test_data.to_csv(f"{output_s3_prefix}/test/test.csv", index=False, header=False)
        df_baseline.to_csv(f"{output_s3_prefix}/baseline/baseline.csv", index=False, header=False)

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()