#!/usr/bin/env python
import logging
import pandas as pd
import os
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Function for training the model
def train_model(X_train, y_train):
    '''
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : pd.DataFrame
        Training data
    y_train : pd.DataFrame
        Labels
    Returns
    -------
    model : 
        Trained machine learning model.
    '''
    # use this logistic regression for training
    logreg = LogisticRegression(C=1.0, 
                                class_weight=None, 
                                dual=False, 
                                fit_intercept=True,
                                intercept_scaling=1, 
                                l1_ratio=None, 
                                max_iter=100,
                                multi_class='auto', 
                                n_jobs=None, 
                                penalty='l2',
                                random_state=0, 
                                solver='liblinear', 
                                tol=0.0001, 
                                verbose=0,
                                warm_start=False
    )
    #assert isinstance(X_train, pd.DataFrame), "Features must be a pandas DataFrame"
    #assert isinstance(y_train, pd.DataFrame), "Targets must be a pandas DataFrame"
    pipe = make_pipeline(
             StandardScaler(), 
             logreg
    )
    # fit the logistic regression to your data
    try:
        pipe.fit(X_train, y_train)
    except ValueError as err:
        logging.info(f"train_model - {err}")
        raise Exception(f"train_model - {err}")
    except AssertionError as msg:
        logging.info(f"train_model - {msg}")
        raise Exception(msg)
    
    return pipe

def store_model(model, model_directory, model_file):
    '''
    store a model in a local directory

    Args:
        model: model to be store
        model_directory: (str) local directory to store the model
        model_file: (str) model file name
    '''
    if not os.path.exists(model_directory):
        logger.info(f"train_lg: {model_directory} not found")
        raise FileNotFoundError(f"train_lg: {model_directory} not found")
    
    model_path = model_directory + "/" + model_file
    with open(model_path, 'wb') as handle:
        dump(model, handle)

def get_model(model_directory, model_file_name):
    '''
    load a model from a local directory

    Args:
        model_file_name: (str) model file name
        model_directory: (str) local directory to get the model from

    Output
        model: trained model
    '''
    if not os.path.exists(model_directory):
        logger.info(f"train_lg: {model_directory} not found")
        raise FileNotFoundError(f"train_lg: {model_directory} not found")
    
    model_file = os.path.join(model_directory, model_file_name)

    if not os.path.exists(model_file):
        logger.info(f"train_lg: {model_file} not found")
        raise FileNotFoundError(f"train_lg: {model_file} not found")
    
    model = load(model_file)
    
    return model