from src.model_build.model_evaluate import score_model
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == '__main__':

    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    test_folder_path = config['test_data_path']
    output_folder_path = config['output_folder_path']
    model_path = config['output_model_path']

    # define test data file
    test_data_file = os.getcwd() + "/" + test_folder_path + "/testdata.csv"
    if not os.path.exists(test_data_file):
        logger.info("Error: {test_data_file} not found")
        exit(1)

    model_file_name = "trainedmodel.pkl"
    # define trained ML model file
    model_file = os.path.join(os.getcwd(),model_path,model_file_name)
    if not os.path.exists(model_file):
        logger.info("Error: {model_file} not found")
        exit(1)

    try:
        f1score = score_model.get_f1_score(model_file_name, model_path, test_data_file)
    except Exception as m:
        logger.info(f"Error: score_model: {m}")
        exit(1)

    f1_score_file = os.path.join(output_folder_path, "latestscore.txt")
    with open(f1_score_file, "w") as f:
        f.write(f1score)
         
