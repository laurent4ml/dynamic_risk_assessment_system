from flask import Flask, session, jsonify, request
from src.model_build.data_split import train_test_split
from src.model_build.model_training import train_lg
import logging
import pandas as pd
import os
import pickle
from sklearn import metrics
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == '__main__':

    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    output_model_path = config['output_model_path']

    split_params ={
        "path": output_folder_path,
        "input_artifact": "finaldata.csv",
        "artifact_root": "churn_data",
        "test_size": 0.2,
        "random_state": 42,
    }
    # Train test split and store them to local directory
    try:
        train_test_split.split_dataset(split_params)
    except Exception as err:
        logger.info(f"Unexpected {err=}, {type(err)=}")

    train_df = pd.read_csv(
        os.getcwd() + "/" + output_folder_path + "/data_split/churn_data_train.csv")
    
    y_train = train_df['exited']
    logger.info(f"y train shape: {y_train.shape}")
    x_train = train_df.drop(['exited'], axis=1)
    logger.info(f"x train shape: {x_train.shape}")
    try:
        model = train_lg.train_model(x_train, y_train)
        logger.info(f"Saving Model")
        # write the trained model to your workspace in a file called trainedmodel.pkl
        with open(output_model_path + '/trainedmodel.pkl', 'wb') as handle:
            pickle.dump(model, handle)

    except Exception as err:
        logger.info(f"Unexpected {err=}, {type(err)=}")


