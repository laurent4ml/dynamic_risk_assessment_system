import os
import logging
import json
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == '__main__':

    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    deployment_folder_path = config['prod_deployment_path']
    output_folder_path = config['output_folder_path']
    model_path = config['output_model_path']

    # latestscore file
    latest_score_file = os.getcwd() + "/" + output_folder_path + "/latestscore.txt"
    if not os.path.exists(latest_score_file):
        logger.info("Error: {latest_score_file} not found")
        exit(1)

    # ingestedfiles file
    ingestedfiles_file = os.getcwd() + "/" + output_folder_path + "/ingestedfiles.txt"
    if not os.path.exists(ingestedfiles_file):
        logger.info("Error: {ingestedfiles_file} not found")
        exit(1)

    # trained ML model file
    model_file = os.getcwd() + "/" + model_path + "/trainedmodel.pkl"
    if not os.path.exists(model_file):
        logger.info("Error: {model_file} not found")
        exit(1)

    destination_directory = os.path.join(os.getcwd(), deployment_folder_path)
    if not os.path.exists(destination_directory):
        logger.info("Error: {destination_directory} not found")
        exit(1)
    
    shutil.copy(latest_score_file, destination_directory)
    logger.info("file {latest_score_file} copied to {destination_directory}")
    shutil.copy(ingestedfiles_file, destination_directory)
    logger.info("file {ingestedfiles_file} copied to {destination_directory}")
    shutil.copy(model_file, destination_directory)
    logger.info("file {model_file} copied to {destination_directory}")
