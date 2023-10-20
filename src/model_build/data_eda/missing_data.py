import pandas as pd
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])

def get_missing_data(data_file):
    '''
    get percentage of missing data per feature

    Input
        data_file (str): data file path
    Ouput
        precents (list): percentage of missing data per numerical feature
    '''
    logger.info(f"missing data: start")
    if not os.path.exists(data_file):
        logger.info(f"Error: {data_file} not found")
        raise FileNotFoundError(f"Error: {data_file} not found")
    
    dataset = pd.read_csv(data_file)
    
    stats = {}
    numerical_features = ("lastmonth_activity", "lastyear_activity", "number_of_employees")
    total_data = len(dataset.index)
    for numerical_feature in numerical_features:
        logger.info(f"Total missing data for {numerical_feature}")
        total_missing_data = dataset[numerical_feature].isna().sum()
        stats[numerical_feature] = total_missing_data / total_data
    
    return stats