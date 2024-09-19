import requests
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.datasets import load_iris
import os
import datetime
import pickle
import uuid
from domino.data_sources import DataSourceClient
import yaml


with open("/mnt/artifacts/DMM_config.yaml") as yamlfile:
    config = yaml.safe_load(yamlfile)

# Your model parmeters
user_name = os.environ['DOMINO_USER_NAME']
external_datasource = config['workbench_datasource_name']
datasourceType = config['datasource']['type']
DMM_datasource_name = config['DMM_datasource_name']
domino_url = config['url']
DMM_model_id = config['external_model_id']
API_key =  os.environ['DOMINO_USER_API_KEY']

# Today's date
date = datetime.datetime.today()
month = date.month
day = date.day
year = date.year

# Today's scoring file name
scoring_file_name = "{}_external_iris_scoring_data_{}_{}_{}.csv".format(user_name,month, day, year)

# Load data for scoring
data = load_iris()
df = pd.DataFrame(data = data['data'], columns = data.feature_names)
df['variety'] = data['target']

scoring_data = df[data.feature_names].copy()

# Jitter the scoring data
for row in scoring_data.iterrows():
    for c in scoring_data.columns:
        scoring_data[c] = np.maximum(0.5, scoring_data[c] + np.random.normal()/25)

# Load the "external" model
file_name = "/mnt/code/models/xgb_iris.pkl"
model = pickle.load(open(file_name, "rb"))

# Get model predictions (numeric)
scoring_data = scoring_data.values.tolist()
model_predictions = model.predict(scoring_data)

# Create the scoring dataset for model moniotring

# Data that was scored, model predictions (as strings), timestamp and event ID for model qulaity monitoring.  
predictions = pd.DataFrame(scoring_data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)' ])
predictions['predictions'] = model_predictions
predictions['variety'] = [data.target_names[y] for y in predictions['predictions']]
predictions.drop('predictions', axis=1, inplace=True)
predictions['timestamp']= datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
event_ids = [uuid.uuid4() for x in range(predictions.shape[0])]
predictions['event_id'] = event_ids

# Save version to Domino Dataset for future reference
predictions.to_csv('/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), scoring_file_name), index=False)

print("Scoring data saved to project's Domino Dataset")

# Time for data to be saved to domino dataset
time.sleep(10)

# A dataset with the scoring data, model predictions and a prediction id are uploaded to the DMM data source (s3 in this example)

# Upload scoring data to DMM data source using a Domino data source (s3 in this example)

print("Uploading to Domino Model Monitoring Datasource")

# instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("{}".format(external_datasource)) # Update

# Upload scoring and ground truth data to monitoring data source
object_store.upload_file(scoring_file_name, '/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), scoring_file_name))
print('{} uploaded to {}'.format(scoring_file_name, DMM_datasource_name))

# Time for data to be saved to monitoring datasource
time.sleep(10)

# Update scoring  file paths with model monitoring API

# This step only updates the file paths, and assumes the external model has already been registered in DMM! See "External_DMM_Quickstart.ipynb"

print('Registering {} from {} data source in DMM'.format(scoring_file_name, external_datasource))

scoring_data_url = "https://{}/model-monitor/v2/api/model/{}/register-dataset/prediction".format(domino_url, DMM_model_id)

# Set up call headers
headers = {
           'X-Domino-Api-Key': API_key,
           'Content-Type': 'application/json'
          }

 
scoring_data_payload = """
{{
    "datasetDetails": {{
            "name": "{0}",
            "datasetType": "file",
            "datasetConfig": {{
                "path": "{0}",
                "fileFormat": "csv"
            }},
            "datasourceName": "{1}",
            "datasourceType": "{2}"
        }}
}}
""".format(scoring_file_name, DMM_datasource_name, datasourceType)
 
# Make api call
scoring_data_response = requests.request("PUT", scoring_data_url, headers=headers, data = scoring_data_payload)
 
# Print response
print(scoring_data_response.text.encode('utf8'))

print("Done!")