import logging
import os
import json
import ingestion
import pandas as pd
import subprocess
from src.model_build.data_preparation import prepare
from src.model_build.data_split import train_test_split
from src.model_build.model_training import train_lg
from src.model_build.model_evaluate import score_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']
test_folder_path = config['test_data_path']

model_file_name = 'trainedmodel.pkl'

logger.info("start deployment process")
# Check and read new data
source_data_files_path = os.path.join(os.getcwd(), input_folder_path)
logger.info("source_data_files_path: {source_data_files_path}")
all_files = ingestion.list_all_files(source_data_files_path)
logger.info(f"list_all_files: {all_files}\n")

# first, read ingestedfiles.txt
#second, determine whether the source data folder has files that aren't listed 
# in ingestedfiles.txt
logger.info("filter_new_files - start")
ingested_file_name = os.path.join(os.getcwd(), prod_deployment_path, "ingestedfiles.txt")
logger.info("filter_new_files - ingested_file_name: {ingested_file_name}")
new_files = ingestion.filter_new_files(ingested_file_name, all_files)
logger.info(f"filter_new_files: {new_files}")

# Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not len(new_files):
    logger.info("no new data files to process - end deployment process\n")
    exit(1)

logger.info("merge_multiple_dataframe - start")
final_data_file = os.path.join(os.getcwd(), output_folder_path, "finaldata.csv")
merge_response = ingestion.merge_multiple_dataframe(
    new_files,
    final_data_file,
    input_folder_path
)
logger.info("merge_multiple_dataframe - end")

if merge_response:
    logger.info("store_new_file - start")
    ingestion.store_new_files(ingested_file_name, new_files)
    logger.info("store_new_file - end")
else:
    logger.info("error merging data")

# Checking for model drift
# Read score from latestscore.txt
latestscore_file_name = os.path.join(os.getcwd(), prod_deployment_path, "latestscore.txt")
latestscore = 0
with open(latestscore_file_name, 'r') as f:
    lines = [line.rstrip() for line in f.readlines()]
    logger.info(f"latestscore - lines: {str(lines)}")
    if len(lines):
        latestscore = lines[0]
logger.info(f"latestscore: {latestscore}")

# train model
clean_params ={
    "path": output_folder_path,
    "input_artifact": "finaldata.csv"
}

# Train test split and store them to local directory file: /data_clean/cleandata.csv
try:
    prepare.clean_data(clean_params)
except Exception as err:
    logger.info(f"training - Unexpected {err=}, {type(err)=}")
    exit(1)

split_data_path = os.path.join(output_folder_path, "data_split")
split_params ={
    "path": output_folder_path + "/data_clean",
    "input_artifact": "cleandata.csv",
    "split_data_path": split_data_path,
    "artifact_root": "churn_data",
    "test_size": 0.2,
    "random_state": 42,
}
 
# Train test split and store them to local directory
try:
    train_test_split.split_dataset(split_params)
except Exception as err:
    logger.info(f"training - Unexpected {err=}, {type(err)=}")
    exit(1)

train_data_file = os.path.join(os.getcwd(), output_folder_path, "data_split/churn_data_train.csv")
if not os.path.exists(train_data_file):
    logger.info(f"training - Error: {train_data_file} not found")
    exit(1)

# load training data
train_df = pd.read_csv(train_data_file)

logger.info(train_df.columns)
logger.info(f"training - train_df type: {type(train_df)}")
logger.info(f"training - train df shape: {train_df.shape}")
y_train = train_df['exited']

logger.info(f"training - y train shape: {y_train.shape}")
logger.info(y_train)
x_train = train_df.drop(['exited'], axis=1)
logger.info(f"training - x train shape: {x_train.shape}")
logger.info(x_train)

try:
    model = train_lg.train_model(x_train, y_train)
except Exception as err:
    logger.info(f"training - Error: {err=}, {type(err)=}")

# write trained model to a file called trainedmodel.pkl
logger.info(f"Saving Model to {model_file_name}")
train_lg.store_model(model, output_model_path, model_file_name)

# define test data file
test_data_file = os.getcwd() + "/" + test_folder_path + "/testdata.csv"
if not os.path.exists(test_data_file):
    logger.info("Error: {test_data_file} not found")
    exit(1)

try:
    f1score = score_model.get_f1_score(model_file_name, output_model_path, test_data_file)
except Exception as m:
    logger.info(f"Error: score_model: {m}")
    exit(1)

#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
# if you found model drift, you should proceed. otherwise, do end the process here
if f1score < latestscore:
    logger.info(f"latest score: {latestscore} > new score {f1score} - end deployement prrocess")
    exit(1)

f1_score_file = os.path.join(output_folder_path, "latestscore.txt")
with open(f1_score_file, "w") as f:
    f.write(f1score)

# Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.run(['python3', 'deployment.py'], capture_output=True)
# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model  
subprocess.run(['python3', 'diagnostics.py'], capture_output=True)
subprocess.run(['python3', 'reporting.py'], capture_output=True)








