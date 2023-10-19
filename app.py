from flask import Flask, session, jsonify, request
import logging
import pandas as pd
from src.model_build.model_prediction import predict
# import predict_exited_from_saved_model
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Set up variables for use in our script
app = Flask(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route("/predictions", methods=['GET','OPTIONS'])
def predictions():
    '''
    Prediction Endpoint returns prediction based on dataset and model

    Output
        predictions (list): predictions from dataset
    '''
    data_file = os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv")

    if not os.path.exists(data_file):
        logger.info(f"predictions: {data_file} not found")
        raise FileNotFoundError(f"predictions: {data_file} not found")
    
    test_data = pd.read_csv(data_file)

    assert isinstance(test_data, pd.DataFrame), "Features must be a pd.DataFrame"

    model_file_name = "trainedmodel.pkl"

    predictions = predict.get_predictions(test_data, model_file_name)

    return str(predictions)

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    # check the score of the deployed model
    return True # add return value (a single F1 score number)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    # check means, medians, and modes for each column
    return True # return a list of all calculated summary statistics

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    # check timing and percent NA values
    return True # add return value for all diagnostics

if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
