from flask import Flask, jsonify, request
import logging
import pandas as pd
from src.model_build.model_prediction import predict
from src.model_build.model_evaluate import score_model
from src.model_build.data_eda import eda, missing_data
from src.model_build.model_timing import timing
from src.diagnostics.dependencies import outdated_packages
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
    logger.info(f"predictions: start")
    model_file_name = "trainedmodel.pkl"
    data_file_name = "finaldata.csv"
    data_file_path = dataset_csv_path
    logger.info(f"predictions: model name {model_file_name}")

    if request.method == 'POST':
        data_file_name = request.form.get('file_name', 'finaldata.csv')
        data_file_path = request.form.get('file_path', dataset_csv_path)

    if not data_file_path or not data_file_name or not len(data_file_path) or not len(data_file_name):
        return jsonify({'input error': 'no file or path'})

    logger.info(f"predictions: data_file_path = {data_file_path}")
    data_file = os.path.join(data_file_path, data_file_name)
    logger.info(f"predictions: file {data_file}")

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
    model_file_name = "trainedmodel_final.pkl"

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
        f1score = score_model.get_f1_score(model_file_name, model_path, test_data_file)
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


@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    diagnostics = {}       
    try:
        timings = timing.execution_time()
        logger.info(timings)
        diagnostics['timings'] = timings
    except Exception as t:
        logger.info(t)
        return jsonify({"error": t})

    data_file = os.path.join(dataset_csv_path, "finaldata.csv")        

    try:
        percents = missing_data.get_missing_data(data_file)
        logger.info(percents)
        diagnostics['missing_data'] = percents
    except Exception as m:
        logger.info(m)
        return jsonify({"error": m})

    try:
        pckgs = outdated_packages.outdated_packages_list()
        logger.info(pckgs)
        diagnostics['outdated_packages'] = pckgs[2:]
    except Exception as m:
        logger.info(m)
        return jsonify({"error": m})

    return jsonify({"diagnostics": diagnostics})

if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
