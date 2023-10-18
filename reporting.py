import pandas as pd
import numpy as np
from sklearn import metrics
import diagnostics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])

def score_model(predicted, actual):
    '''
    calculate a confusion matrix using the test data and the deployed model

    Args
        preds (DataFrame): predictions generated based on test data
        actual (DataFrame): actual response to validate
    '''
    logger.info("score model: start")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    # write the confusion matrix to the workspace
    df_cfm = pd.DataFrame(confusion_matrix)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True)
    
    cm_file = os.path.join(output_model_path, 'confusionmatrix.png')
    cfm_plot.figure.savefig(cm_file)

if __name__ == '__main__':
    test_file = os.path.join(test_data_path, "testdata.csv")
    if not os.path.exists(test_file):
        logger.info(f"Error: {test_file} not found")
        exit(1)

    test_data = pd.read_csv(test_file)
    y_test = test_data['exited']
    test_data = test_data.drop(['corporation','exited'], axis=1)
    preds = diagnostics.model_predictions(test_data)
    logger.info(preds)

    score_model(preds, y_test)
