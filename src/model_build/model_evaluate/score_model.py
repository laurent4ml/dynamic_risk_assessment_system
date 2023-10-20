#!/usr/bin/env python
import logging
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.model_build.model_training import train_lg

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Function for training the model
def get_f1_score(model_file_name, model_path, test_data_file):
    '''
    Calculate the F1 score of your trained model on your testing data
    Write the F1 score to a file called latestscore.txt
    
    Args:
        model_file_name: (str) logistic regression trained model file name
        model_path: (str) logistic regression trained model file path
        test_data: (str) test file name with path

    Output:
        f1_score: (float) f1 score returned by model on test data
    '''
    if not os.path.exists(test_data_file):
        logger.info("Error: {test_data_file} not found")
        raise FileNotFoundError("scored_model: {test_data_file} not found")
    logger.info(f"scored_model - loading file: {test_data_file}")
    test_data = pd.read_csv(test_data_file)

    assert isinstance(test_data, pd.DataFrame), "Features must be a pd.DataFrame"

    X_test = test_data.drop(['exited','corporation'], axis=1)

    y_test = test_data['exited']

    model_file = os.path.join(os.getcwd(), model_path, model_file_name)
    if not os.path.exists(model_file):
        logger.info(f"scored_model: {model_file} not found")
        raise FileNotFoundError("scored_model: {model_file} not found")
    
    logger.info(f"scored_model - loading model file: {model_file}")
    
    lr_model = train_lg.get_model(model_path, model_file_name)

    logger.info(f"scored_model - type of lr_model: {type(lr_model)}")

    preds = lr_model.predict(X_test)

    logger.info("scored_model - Evaluating overall model performance")
    metrics = precision_recall_fscore_support(y_test, preds, 
                                                            average="micro")
    precision = metrics[0]
    recall = metrics[1]
    fbeta = metrics[2]
    logger.info(f"scored_model - precision: {precision}")
    logger.info(f"scored_model - recall: {recall}")
    logger.info(f"scored_model - fbeta: {fbeta}")
    return str(fbeta)