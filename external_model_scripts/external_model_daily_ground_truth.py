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
scoring_file_name = "external_iris_scoring_data_{}_{}_{}.csv".format(month, day, year)

# Today's ground truth file name
gt_file_name = "external_iris_ground_truth_{}_{}_{}.csv".format(month, day, year)

# Create the "dummy" ground truth dataset

predictions = pd.read_csv('/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), scoring_file_name))

ground_truth = pd.DataFrame(columns=['event_id', 'iris_ground_truth'])
ground_truth['event_id'] = predictions['event_id']
ground_truth['iris_ground_truth'] = predictions['variety']

# These row labels help find some diferent iris types in our initial scoring data
end_index = predictions.shape[0]
mid_index = int(round(predictions.shape[0] / 2, 0))

# Simulate some classifcation errors. This makes our confusion matrix interesting.
ground_truth.iloc[0, 1] = 'virginica'
ground_truth.iloc[1, 1] = 'versicolor'
ground_truth.iloc[mid_index-1, 1] = 'versicolor'
ground_truth.iloc[mid_index, 1] = 'virginica'
ground_truth.iloc[end_index-2, 1] = 'setosa'
ground_truth.iloc[end_index-1, 1] = 'setosa'

# Save each version locally 
ground_truth.to_csv('/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), gt_file_name), index=False)

print("Ground truth data saved to project's Domino Dataset")

# Time for data to be saved to domino dataset
time.sleep(10)

print("Uploading ground truth data to Domino Model Monitoring Datasource")

# instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("{}".format(external_datasource))

object_store.upload_file(gt_file_name, '/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), gt_file_name))
print('{} uploaded to {}'.format(gt_file_name, DMM_datasource_name))

# Time for data to be saved to external datasource
time.sleep(10)

# Update ground truth file paths with model monitoring API

# This step only updates the file paths, and assumes the external model has already been registered in DMM! See "External_DMM_Quickstart.ipynb"

print('Registering {} from {} data source in DMM'.format(gt_file_name, external_datasource))

ground_truth_data_url = "https://{}/model-monitor/v2/api/model/{}/register-dataset/ground_truth".format(domino_url, DMM_model_id)

# Set up call headers
headers = {
           'X-Domino-Api-Key': API_key,
           'Content-Type': 'application/json'
          }

 
ground_truth_data_payload = """
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
""".format(gt_file_name, DMM_datasource_name, datasourceType)
 
# Make api call
ground_truth_data_response = requests.request("PUT", ground_truth_data_url, headers=headers, data = ground_truth_data_payload)
 
# Print response
print(ground_truth_data_response.text.encode('utf8'))

print("Done!")

