import domino
import numpy as np
import pandas as pd
import datetime
import rfc3339
from domino.training_sets import TrainingSetClient, model

# Connect to client
dmm_model_id = '657922ab9475d8d2f34562cf'
d = domino.Domino("dave_heinicke/Custom-Metric-Example")
metrics_client = d.custom_metrics_client()

# Define time window
startDate = datetime.datetime.today() - datetime.timedelta(days=365)
startDate = rfc3339.rfc3339(startDate)
endDate = rfc3339.rfc3339(datetime.datetime.today())


# Define your metric
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

# Read in data to measue


# Column we want to calculate metric for
drift_column_name = 'petal length (cm)'

# Print existing Training Sets in this Project

training_set = TrainingSetClient.get_training_set_version(
    training_set_name = "iris_python_multi_classification",
    number=1
    )

training_df = training_set.load_training_pandas()

train = training_df[drift_column_name]

scoring_data = pd.read_csv('/domino/datasets/local/Custom-Metric-Example/iris_scoring_data_2023-12-18.csv')

inference = scoring_data[drift_column_name]

# Calculate the metric
hellinger_distance = hellinger_distance(train, inference)
print('Hellinger distance between scoring and traiing data is: {}'.format(str(round(hellinger_distance, 3))))

# Log metric & timestamp
timestamp = rfc3339.rfc3339(datetime.datetime.now()) # datetime.datetime.now().isoformat()
metrics_client.log_metric(dmm_model_id, "hellinger_distance", hellinger_distance, timestamp, { "Column" : drift_column_name})
