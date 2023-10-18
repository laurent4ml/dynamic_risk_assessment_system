import pandas as pd
import timeit
import os
import json
import logging
import subprocess
from src.model_build.model_training import train_lg

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_folder_path = os.path.join(config['output_folder_path'])

def model_predictions(dataset):
    '''
    Function to get model predictions
    read the deployed model and a test dataset, calculate predictions

    Input
        dataset (DataFrame): dataset to use to get predictions
    Output
        predictions (list): predictions based on trained model and test data
    '''
    # trained ML model file
    logger.info("model_predictions: start")
    model_file_name = "trainedmodel.pkl"
    model_file = os.path.join(prod_deployment_path, model_file_name)
    if not os.path.exists(model_file):
        logger.info(f"Error: {model_file} not found")
        exit(1)

    model = train_lg.get_model(prod_deployment_path, model_file_name)
    
    predictions = model.predict(dataset)
    logger.info(predictions)
    return predictions

# Function to get summary statistics
def dataframe_summary():

    data_file = os.path.join(output_folder_path, "finaldata.csv")
    if not os.path.exists(data_file):
        logger.info(f"Error: {data_file} not found")
        exit(1)
    
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

def missing_data():
    '''
    get percentage of missing data per feature

    Ouput
        precents (list): percentage of missing data per numerical feature
    '''
    logger.info(f"missing data: start")
    data_file = os.path.join(output_folder_path, "finaldata.csv")
    if not os.path.exists(data_file):
        logger.info(f"Error: {data_file} not found")
        exit(1)
    
    dataset = pd.read_csv(data_file)
    
    stats = {}
    numerical_features = ("lastmonth_activity", "lastyear_activity", "number_of_employees")
    total_data = len(dataset.index)
    for numerical_feature in numerical_features:
        logger.info(f"Total missing data for {numerical_feature}")
        total_missing_data = dataset[numerical_feature].isna().sum()
        stats[numerical_feature] = total_missing_data / total_data
    
    return stats

# Function to get timings
def execution_time():
    '''
    calculate timing of training.py and ingestion.py

    Output
        timings (list): timing for training and ingestion tasks
    '''
    timings = {}
    timings['ingestion'] = ingestion_timing()
    timings['training'] = training_timing()
    return timings

def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    return timing

# Function to check dependencies
def outdated_packages_list():
    ''' get list of outdated packages'''
    outdated_packages = subprocess.check_output(['pip', 'list','--outdated'])
    with open('outdated.txt', 'wb') as f:
        f.write(outdated_packages)

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
    preds = model_predictions(test_data)
    logger.info(preds)
    stats = dataframe_summary()
    logger.info(stats)
    percents = missing_data()
    logger.info(percents)
    timings = execution_time()
    logger.info(timings)
    outdated_packages = outdated_packages_list()
    logger.info(outdated_packages)
    write_dependencies()





    
