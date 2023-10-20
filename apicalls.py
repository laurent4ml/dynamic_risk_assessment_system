import requests
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])

# Specify a API URL
URL = "http://127.0.0.1"

#Call each API endpoint and store the responses
headers = {'content-type' : 'application/json'}
payload = {"file_path": "testdata", "file_name": "testdata.csv"}
params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103'}
response1 = requests.post(URL + ':5000/predictions', 
                        params=params, 
                        data=json.dumps(payload))
if not response1.status_code == 200:
    logging.info(f"Error Response 1: {response1.text}")
    exit(1)

if response1.text == '{"input error":"no file or path"}':
    logging.info(f"Input Error: no file or path")
    exit(1)

response_content1 = json.loads(response1.text)
logging.info(f"response1: {response1.text}")
logging.info(f"Response 1 - type: {type(response_content1)}")

response2 = requests.get(URL + ':5000/scoring')
if not response2.status_code == 200:
    logging.info(f"Error Response 2 - scoring: {response2.text}")
    exit(1)

logging.info(f"Response 2 - scoring: {response2.text}")

response_content2 = json.loads(response2.text)
logging.info(f"Response 2 - type: {type(response_content2)}")
#merged_dict1 = dict(response_content1.items() | response_content2.items())
logging.info(f"Response 3 start")
response3 = requests.get(URL + ':5000/diagnostics')
if not response3.status_code == 200:
    logging.info(f"Error Response 3: {response3.text}")
    exit(1)

if len(response3.text) == 0:
    exit(1)

response_content3 = json.loads(response3.text)
logging.info(f"Response 3 - type: {type(response_content3)}")
#merged_dict2 = dict(merged_dict1.items() | response_content3.items())

response4 = requests.get(URL + ':5000/summarystats')
if not response4.status_code == 200:
    logging.info(f"Error Response 4: {response4.text}")
    exit(1)

if len(response4.text) == 0:
    exit(1)
logging.info(f"response4: {response4.text}")
response_content4 = json.loads(response4.text)
logging.info(f"Response 4 - type: {type(response_content4)}")
# responses = dict(response_content2.items() | response_content2.items() | response_content3.items() | response_content4.items())
responses = {**response_content2, **response_content2, **response_content3, **response_content4}

logging.info(f"responses: {responses}")
response_file = os.path.join(os.getcwd(), output_model_path, 'apireturns.txt')
with open(response_file , 'w') as f:
    f.write(str(responses))
