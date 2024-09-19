import domino
import numpy as np
import pandas as pd
import datetime
from domino.training_sets import TrainingSetClient, model

dmm_model_id = '65553e919475d8d2f345628d'
d = domino.Domino("dave_heinicke/Custom-Metric-Example")
metrics_client = d.custom_metrics_client()


# Column we want to calculate metric for
drift_column_name = 'petal length (cm)'

# Definition of the custom metric
def hellinger_distance(train, inference):
    
    # distance between training data and inference data
    # train is the ditribution of an input feature in the training data
    # inference is the dsitribution of a feature being sent to the model API
    
    n = min(len(train), len(inference))
    sum = 0.0
    
    for i in range(n):
        sum += (np.sqrt(train[i]) - np.sqrt(inference[i]))**2
        
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    
    return result

# Calculate the Custom Metric, comparing scoring data to original training data

training_set = TrainingSetClient.get_training_set_version(
    training_set_name = "iris_python_multi_classification",
    number=1
    )

training_df = training_set.load_training_pandas()
train = training_df[drift_column_name]

# Fetch inference data 

# scoring_data = pd.read_csv('/domino/datasets/local/Custom-Metric-Example/iris_scoring_data_2023-12-18.csv')

# inference = scoring_data[drift_column_name]

# Fetch ingested predictions from Domino Dataset
date = datetime.datetime.today()
month = date.month
day = date.day

# Manually enter dates
# month = 12
# day = 14

# Parquet file path(s)
path = '/domino/datasets/local/prediction_data/657920c188931a7b02098e3a/$$date$$=2023-{}-{}Z/'.format(month, day)

paths = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".parquet"):
             paths.append(os.path.join(root, file))
                
predictions = pd.DataFrame(columns=['petal width (cm)', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'variety', 'timestamp', 'event_id','__domino_timestamp'])

for file in paths:
    predictions = pd.concat([predictions, pd.read_parquet(file)], axis=0)

# Get the series we're using for the custom metric
inference = predictions[drift_column_name]

# Calculate the metric
hellinger_distance = hellinger_distance(train, inference)
print('Hellinger distance between scoring and traiing data is: {}'.format(str(round(hellinger_distance, 3))))

timestamp = "2023-12-17T00:00:00Z"

metrics_client.log_metric(dmm_model_id, "hellinger_distance", hellinger_distance, timestamp, { "Column" : drift_column_name})