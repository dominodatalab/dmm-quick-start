import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.datasets import load_iris
import os
import datetime
import yaml

with open("/mnt/artifacts/DMM_config.yaml") as yamlfile:
    config = yaml.safe_load(yamlfile)
user_name = os.environ['DOMINO_USER_NAME']

data = load_iris()

df = pd.DataFrame(data = data['data'], columns = data.feature_names)
df['variety'] = data['target']

scoring_data = df[data.feature_names].copy()

# Jitter the scoring data
for row in scoring_data.iterrows():
    for c in scoring_data.columns:
        scoring_data[c] = np.maximum(0.1, scoring_data[c] + np.random.normal()/25)
        
# Save in Domino for future reference

date = str(datetime.datetime.today()).split()[0]

# Retreive ingested predictions from Domino Dataset
date = datetime.datetime.today()
month = date.month
day = date.day
year = date.year

scoring_data.to_csv('/mnt/data/{}/{}_external_iris_scoring_data_{}_{}_{}.csv'.format(os.environ.get('DOMINO_PROJECT_NAME'),user_name, month, day, year), index=False)

scoring_data = scoring_data.values.tolist()

model_url = config['integrated_model_url']
model_auth_token = config['integrated_model_auth']

response = requests.post(model_url,
    auth=(
        model_auth_token,
        model_auth_token
    ),
    json={
        "data": scoring_data
    }
)

    
print(response.status_code)
print(response.headers)
print(response.json())