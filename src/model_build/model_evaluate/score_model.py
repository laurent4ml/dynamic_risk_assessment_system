#!/usr/bin/env python
import logging
import os
import pandas as pd
from joblib import load
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Function for training the model
def get_f1_score(model_file, test_data_file):
    '''
    Calculate the F1 score of your trained model on your testing data
    Write the F1 score to a file called latestscore.txt
    
    Args:
        model_file: (str) logistic regression trained model file path and name
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

    # Read in your trained ML model from the directory specified in the 
    if not os.path.exists(model_file):
        logger.info("scored_model: {model_file} not found")
        raise FileNotFoundError("scored_model: {model_file} not found")
    logger.info(f"scored_model - loading model file: {model_file}")
    with open(model_file , 'rb') as f:
        lr_model = load(f)

    # raise AssertionError(f"scored_model error loading model: {msg}")
    logger.info(f"scored_model - type of lr_model: {type(lr_model)}")
    #logger.info(f"scored_model - lr_model.coef_: {lr_model.coef_}")
    # logger.info(f"scored_model - len(lr_model.coef_[0]): {len(lr_model.coef_[0])}")

    # assert len(lr_model.coef_[0]) == X_test.shape[1], "scored_model: Data dimension incorrect"

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
    
