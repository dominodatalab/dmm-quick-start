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

# Retrieve the metrics
try:
    metrics = metrics_client.read_metrics(dmm_model_id, "hellinger_distance", startDate, endDate)
    
except Exception as err:
    logging.error("Unable to fetch metrics")
    raise err
    
print(metrics)