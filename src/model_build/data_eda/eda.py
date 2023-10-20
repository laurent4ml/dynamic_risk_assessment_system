#!/usr/bin/env python
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def dataframe_summary(data_file):
    '''
    Function to get summary statistics

    Input
        data_file: (str) data file path

    Output
        stats (list) stats for data file
    '''
    if not os.path.exists(data_file):
        logger.info(f"Error: {data_file} not found")
        raise FileNotFoundError(f"Error: {data_file} not found")
    
    dataset = pd.read_csv(data_file)

    stats = {}

    numerical_features = ("lastmonth_activity", "lastyear_activity", "number_of_employees")
    for numerical_feature in numerical_features:
        stat = {}
        logger.info(f"Log descriptive stats for {numerical_feature}")
        mean = dataset[numerical_feature].mean()
        stat['mean'] = mean
        logger.info(f"mean for {numerical_feature}: {mean}")
        stddev = dataset[numerical_feature].std()
        stat['std'] = stddev
        logger.info(f"stddev for {numerical_feature}: {stddev}")
        median = dataset[numerical_feature].median()
        stat['median'] = median
        logger.info(f"median for {numerical_feature}: {median}")
        stats[numerical_feature] = stat
    return stats
