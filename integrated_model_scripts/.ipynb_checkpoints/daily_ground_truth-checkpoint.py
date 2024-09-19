import pandas as pd
import json
import os
import requests
import datetime
import time
import yaml

with open("/mnt/artifacts/DMM_config.yaml") as yamlfile:
    config = yaml.safe_load(yamlfile)

# Your model parmeters
external_datasource = config['workbench_datasource_name']
datasourceType = config['datasource']['type']
DMM_datasource_name = config['DMM_datasource_name']
domino_url = config['url']
DMM_model_id = config['integrated_model_id']
API_key =  os.environ['DOMINO_USER_API_KEY']
prediction_data_dir = config['prediction_data_dir']


# Retreive ingested predictions from Domino Dataset
date = datetime.datetime.today()
month = date.month
day = date.day
year = date.year

# Today's scoring file name
scoring_file_name = "external_iris_scoring_data_{}_{}_{}.csv".format(month, day, year)

# Today's ground truth file name
gt_file_name = "external_iris_ground_truth_{}_{}_{}.csv".format(month, day, year)

# Optional - manually enter dates for manual ingest of a specific date
# month = 2
# day = 1
# year = 2024

# Parquet file path(s)
path = '/mnt/data/prediction_data/{}/$$date$$={}-{:02d}-{:02d}Z/'.format(prediction_data_dir, year, month, day)

print(path)

paths = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".parquet"):
             paths.append(os.path.join(root, file))
                
print(paths)

predictions = pd.DataFrame(columns=['petal width (cm)', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'variety', 'timestamp', 'event_id','__domino_timestamp'])

for file in paths:
    predictions = pd.concat([predictions, pd.read_parquet(file)], axis=0)

print(predictions.shape)


# Create dummy ground truth data
event_id = predictions['event_id']
iris_ground_truth = predictions['variety']

# Create a new dataframe
ground_truth = pd.DataFrame(columns=['event_id', 'iris_ground_truth'])
ground_truth['event_id'] = event_id
ground_truth['iris_ground_truth'] = iris_ground_truth

print(ground_truth.shape)

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
# Time for data to be saved to domino dataset
time.sleep(10)

print("Ground truth data saved to project's Domino Dataset")

from domino.data_sources import DataSourceClient

# instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("{}".format(external_datasource))

object_store.upload_file(gt_file_name, '/mnt/data/{}/{}'.format(os.environ.get('DOMINO_PROJECT_NAME'), gt_file_name))
# Time for data to be saved to external datasource
time.sleep(10)

print('{} uploaded to {}'.format(gt_file_name, DMM_datasource_name))

ground_truth_url = "https://{}/model-monitor/v2/api/model/{}/register-dataset/ground_truth".format(domino_url, DMM_model_id)

print('Registering {} From {} data source in DMM'.format(gt_file_name, external_datasource))
 
# create GT payload    
 
# Set up call headers
headers = {
           'X-Domino-Api-Key': API_key,
           'Content-Type': 'application/json'
          }

 
ground_truth_payload = """
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
ground_truth_response = requests.request("PUT", ground_truth_url, headers=headers, data = ground_truth_payload)
 
# Print response
print(ground_truth_response.text.encode('utf8'))
 
print('DONE!')