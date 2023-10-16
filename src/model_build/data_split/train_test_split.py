#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def split_dataset(args):
    """
    Getting finaldata dataset and splits it into train and test datasets.
    Store the train and test datasets on local.

    Args:
        - path (str): input file path
        - split_data_path (str): folder to store split datasets
        - input_artifact (str): Input artifact string
        - artifact_root (str): Artifact root
        - artifact_type (str): Artifact type
        - test_size (int): Test size
        - random_state  (int): Random state
    """
    artifact = args['input_artifact']
    directory = args['path']
    split_data_directory = args['split_data_path']
    test_size = args['test_size']
    random_state = args['random_state']
    artifact_root = args['artifact_root']
 
    artifact_path = f"{directory}/{artifact}"
    logger.info(f"split_dataset - downloading and reading artifact: {artifact_path}")
    # Read in finaldata.csv using the pandas module.
    finaldata = pd.read_csv(os.getcwd() + "/" + artifact_path, low_memory=False)

    # Split model_dev/test
    logger.info("split_dataset - splitting data into train and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        finaldata,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info("split_dataset - end splitting data")
    for split, df in splits.items():

        # Make the artifact name from the provided root plus the name of the
        # split
        artifact_name = f"{artifact_root}_{split}.csv"

        if not os.path.exists(split_data_directory):
            logger.info(f"split_dataset - Creating {split_data_directory}")
            os.mkdir(split_data_directory)

        path = os.path.join(split_data_directory, artifact_name)
        logger.info(f"split_dataset - Saving the {split} dataset to {path}")
        # Save to local filesystem
        df.to_csv(path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--path",
        type=str,
        help="File path",
        required=True,
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
        "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
        "splitting",
        type=int,
        required=False,
        default=42,
    )

    args = parser.parse_args()

    split_dataset(args)