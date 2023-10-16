#!/usr/bin/env python
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def clean_data(args):
    '''
    clean datasets, removing categorical column 'corporation' and droping duplicates

    Input:
        path (str): input file path
        input_artifact (str): Input artifact string
    '''
    artifact = args['input_artifact']
    path = args['path']

    if not os.path.exists(path):
        logger.info("prepare: {input_path} not found")
        raise FileNotFoundError("prepare: {input_path} not found")
    
    artifact_path = f"{path}/{artifact}"
    logger.info(f"clean_data -  artifact: {artifact_path}")

    # Read in finaldata.csv using the pandas module.
    data = pd.read_csv(os.getcwd() + "/" + artifact_path, low_memory=False)
    
    assert isinstance(data, pd.DataFrame), "prepare - data not a pandas dataframe"

    logger.info(f"clean_data - data shape: {data.shape}")
    data = data.drop(['corporation'], axis=1)
    logger.info(f"clean_data - data shape after drop column: {data.shape}")
    data = data.drop_duplicates()
    logger.info(f"clean_data - data shape after drop duplicates: {data.shape}")
    
    local_directory = os.path.join(path, "data_clean")

    if not os.path.exists(local_directory):
        logger.info(f"prepare - Creating Dirrectory: {local_directory}")
        os.mkdir(local_directory)

    output_file = os.getcwd() + "/" + local_directory + "/cleandata.csv"
    logger.info(f"clean_data - output_file: {output_file}")
    with open(output_file, "wb") as f:
        try:
            data.to_csv(f, index=False)
        except Exception as e:
            logger.info(f"clean_data - writing to file: {e}")
            raise Exception(f"clean_data - writing to file: {e}")