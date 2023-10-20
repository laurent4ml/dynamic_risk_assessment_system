from src.model_build.data_preparation import prepare
from src.model_build.data_split import train_test_split
from src.model_build.model_training import train_lg
import logging
import pandas as pd
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == '__main__':

    model_file = 'trainedmodel.pkl'

    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f)

    folder_path = config['output_folder_path']
    output_model_path = config['output_model_path']

    clean_params ={
        "path": folder_path,
        "input_artifact": "finaldata.csv"
    }

    # Train test split and store them to local directory
    try:
        prepare.clean_data(clean_params)
    except Exception as err:
        logger.info(f"training - Unexpected {err=}, {type(err)=}")
        exit(1)

    split_data_path = os.path.join(folder_path, "data_split")
    split_params ={
        "path": folder_path + "/data_clean",
        "input_artifact": "cleandata.csv",
        "split_data_path": split_data_path,
        "artifact_root": "churn_data",
        "test_size": 0.2,
        "random_state": 42,
    }
    
    # Train test split and store them to local directory
    try:
        train_test_split.split_dataset(split_params)
    except Exception as err:
        logger.info(f"training - Unexpected {err=}, {type(err)=}")
        exit(1)

    train_data_file = os.getcwd() + "/" + folder_path + "/data_split/churn_data_train.csv"
    if not os.path.exists(train_data_file):
        logger.info(f"training - Error: {train_data_file} not found")
        exit(1)

    # load training data
    train_df = pd.read_csv(train_data_file)
    logger.info(f"training - train_df shape: {train_df.shape}")
    # simple data transformation
    logger.info(train_df.columns)
    train_df = train_df.drop(['Unnamed: 0'],axis=1)
    logger.info(train_df.columns)
    logger.info(f"training - train_df type: {type(train_df)}")
    logger.info(f"training - train df shape: {train_df.shape}")
    y_train = train_df['exited']
    
    logger.info(f"training - y train shape: {y_train.shape}")
    logger.info(y_train)
    x_train = train_df.drop(['exited'], axis=1)
    logger.info(f"training - x train shape: {x_train.shape}")
    logger.info(x_train)
    # train model
    try:
        model = train_lg.train_model(x_train, y_train)
        # write trained model to a file called trainedmodel.pkl
        model_directory = os.getcwd() + "/" + output_model_path
        logger.info(f"Saving Model to {model_file}")
        train_lg.store_model(model, model_directory, model_file)
    except Exception as err:
        logger.info(f"training - Error: {err=}, {type(err)=}")



