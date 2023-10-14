#!/usr/bin/env python
import logging
import numpy as np
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
    X_train : np.array
        Training data
    y_train : np.array
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
                                multi_class='warn', 
                                n_jobs=None, 
                                penalty='l2',
                                random_state=0, 
                                solver='liblinear', 
                                tol=0.0001, 
                                verbose=0,
                                warm_start=False
    )
    
    try:
        assert isinstance(X_train, np.ndarray), "Features must be a Numpy array"
        assert isinstance(y_train, np.ndarray), "Targets must be a Numpy array"
        pipe = make_pipeline(
            StandardScaler(), logreg
        )
        # fit the logistic regression to your data
        pipe.fit(X_train, y_train)
        return pipe
    except AssertionError as msg:
        return msg