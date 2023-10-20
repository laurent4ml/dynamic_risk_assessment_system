import pandas as pd
import os
import json
import logging
import subprocess
from src.model_build.model_prediction import predict
from src.model_build.data_eda import eda, missing_data
from src.model_build.model_timing import timing
from src.diagnostics.dependencies import outdated_packages

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 
 
test_data_path = os.path.join(config['test_data_path'])
output_folder_path = os.path.join(config['output_folder_path'])     

def write_dependencies():
    ''' 
    checks the current and latest versions of all the modules 
    that your scripts use
    '''
    requirements = subprocess.check_output(['pip', 'freeze'])

    with open('requirements.txt', 'wb') as f:
        f.write(requirements)


if __name__ == '__main__':
    test_file = os.path.join(test_data_path, "testdata.csv")
    if not os.path.exists(test_file):
        logger.info(f"Error: {test_file} not found")
        exit(1)

    test_data = pd.read_csv(test_file)
    test_data = test_data.drop(['corporation','exited'], axis=1)
    preds = predict.get_predictions(test_data, "trainedmodel.pkl")
    logger.info(preds)

    data_file = os.path.join(output_folder_path, "finaldata.csv")
    if not os.path.exists(data_file):
        logger.info(f"Error: {data_file} not found")
        exit(1)
     
    try:
        stats = eda.dataframe_summary(data_file)
        logger.info(stats)
    except Exception as m:
        logger.info(m)
    
    try:
        percents = missing_data.get_missing_data(data_file)
        logger.info(percents)
    except Exception as m:
        logger.info(m)

    try:
        timings = timing.execution_time()
        logger.info(timings)
    except Exception as t:
        logger.info(t)
    
    outdated_packages = outdated_packages.outdated_packages_list()
    logger.info(outdated_packages)
    write_dependencies()
