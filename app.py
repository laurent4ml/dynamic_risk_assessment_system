from flask import Flask, jsonify, request
import logging
import pandas as pd
from src.model_build.model_prediction import predict
from src.model_build.model_evaluate import score_model
from src.model_build.data_eda import eda
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Set up variables for use in our script
app = Flask(__name__)

with open('config.json','r') as f:
    config = json.load(f)

test_folder_path = config['test_data_path']
model_path = config['output_model_path']
dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route("/predictions", methods=['GET','POST','OPTIONS'])
def predictions():
    '''
    Prediction Endpoint returns prediction based on dataset and model

    Output
        predictions (list): predictions from dataset
    '''
    model_file_name = "trainedmodel.pkl"
    data_file_name = "finaldata.csv"

    if request.method == 'POST':
        data_file_name = request.form['file']

    data_file = os.path.join(os.getcwd(), dataset_csv_path, data_file_name)

    if not os.path.exists(data_file):
        logger.info(f"predictions: {data_file} not found")
        return jsonify({"input error": f"'{data_file_name}' file not found"})
    
    test_data = pd.read_csv(data_file)

    assert isinstance(test_data, pd.DataFrame), "Features must be a pd.DataFrame"

    predictions = predict.get_predictions(test_data, model_file_name)
    preds = predictions.tolist()
    logger.info(f"predictions: {preds}")
    return jsonify({"predictions": preds})

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():    
    test_data_file_name = "testdata.csv"
    model_file_name = "trainedmodel.pkl"

    # define test data file
    test_data_file = os.path.join(os.getcwd(),test_folder_path,test_data_file_name)
    if not os.path.exists(test_data_file):
        logger.info("Error: {test_data_file_name} not found")
        return jsonify({"error": f"'{test_data_file_name}' file not found"})

    # define trained ML model file
    model_file = os.path.join(os.getcwd(),model_path,model_file_name)
    if not os.path.exists(model_file):
        logger.info("Error: {model_file_name} not found")
        return jsonify({"error": f"'{model_file_name}' file not found"})

    try:
        f1score = score_model.get_f1_score(model_file, test_data_file)
    except Exception as m:
        logger.info(f"Error: score_model: {m}")
        return jsonify({"error": m})

    return jsonify({"F1 score": f1score})

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    stats = []
    
    data_file = os.path.join(dataset_csv_path, "finaldata.csv")        

    try:
        stats = eda.dataframe_summary(data_file)
        logger.info(stats)
    except Exception as m:
        logger.info(m)
        return jsonify({"error": m})

    return jsonify({"Stats": stats})

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    # check timing and percent NA values
    return True # add return value for all diagnostics

if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
