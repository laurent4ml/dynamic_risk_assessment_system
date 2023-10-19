#!/usr/bin/env python
import logging
import os
import json
from src.model_build.model_training import train_lg

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Function for training the model
def get_predictions(dataset, model_file_name = "trainedmodel.pkl"):
    '''
    Function to get model predictions
    read the deployed model and a test dataset, calculate predictions

    Input
        dataset (DataFrame): dataset to use to get predictions
        model_file (str): train model path
    Output
        predictions (list): predictions based on trained model and test data
    '''
    logger.info("model_predictions: start")
    predictions = None
    with open('config.json','r') as f:
        config = json.load(f)

        prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path']) 
        model = train_lg.get_model(prod_deployment_path, model_file_name)

        if 'exited' in dataset:
            logger.info("model_predictions: dropping column 'exited'")
            dataset = dataset.drop(['exited'], axis=1)
        if 'corporation' in dataset:
            logger.info("model_predictions: dropping column 'corporation'")
            dataset = dataset.drop(['corporation'], axis=1)
        logger.info(f"dataset shape: {dataset.shape}")
        logger.info(f"dataset columns: {dataset.columns}")
        if len(dataset.columns) > 3:
            dataset = dataset.drop(['Unnamed: 0'], axis=1)
        try:
            predictions = model.predict(dataset.values)
            logger.info(predictions)
        except ValueError as v:
            logger.info(f"Error {v}")
    return predictions