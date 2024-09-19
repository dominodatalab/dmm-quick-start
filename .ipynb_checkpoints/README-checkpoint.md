# DMM-Quick-Setup

This repo contains tutorial notebooks and setup scripts to build an initial example for Domino Model Monitoring.

<p align="center">
<img src = https://github.com/dominodatalab/dmm-quick-start/blob/main/readme_images/Iris_Overview.png width="800">
</p>


## Background

These examples help create starter models in Domino Model Monitoring (DMM). 

Domino Model Monitoring can monitor Domino Model APIs (using Integrated Model Monitoring) or external models. External models include models run in batch as Domino jobs and models hosted outside of Domino, such as in Sagemaker or on-prem.

## Initial Setup

First step is to set up an external monitoring data source, and connect it to both Domino Model Monitoring and your Domino Project.

The initial setup Notebook **1_Initial_Setup.ipynb** walks through these steps, and creates a config file for accessing that external data source that will be used in the other three notebooks.

## External Model Monitoring Example (Using a Domino Job)

To get started with external model monitoring, begin with **2_Integrated_DMM_Quickstart.ipynb**

External models assume the model is already trained and being hosted somewhere other than a Domino Model API. 

The high level steps for external model monitoring are:

### I. Set Up your Model Monitoring Data Source

(1) Train a model in Domino's Workbench. 

(2) Register an external datasource in both Domino Model Monitoring and in the Workbench.

(3) Upload the training dataset used to train the model to the external datasource.

(4) Register your model in Domino Model Monitoring, providing Domino with a path to your training dataset.


### II. Capture Data Drift

(5) Score some data in batch using a Domino Job.

(6) Upload scoring data and predictions from your model to the external datasource. 

(7) Provide Domino Model monitoring the path to your scoring data file(s) (either in the UI or via API). 

(8) Schedule drift data checks in Domino Model Monitoring.

### III. (Optional) Capture ground truth labels for Model Quality monitoring

(9) Upload a dataset with ground truth labels to the external datasource created in Step 2.

(10) Provide Domino Model monitoring the path to your ground truth data file(s) (either in the UI or via API). 

(11) Schedule ground truth monitoring checks in Domino Model Monitoring.


## Integrated Model Monitoring Example (Using a Domino Model API)

To get started with integrated monitoring, begin with **3_Integrated_DMM_Quickstart.ipynb**

The high level steps for integrated monitoring are:

### I. Train Your Model

(1) Train a model. While not required for monitoring, it is best practice to register the model in Domino's Model 
Catalog for documentation of model versions, approvals, and artifacts. For integrated monitoring, be sure that the 
model invokes Domino's [DataCaptureClient](https://docs.dominodatalab.com/en/latest/user_guide/93e5c0/set-up-prediction-capture/
) so that Domino can automatically capture inference data.

(2) Register the data used to train that model as a [Training Dataset](https://docs.dominodatalab.com/en/latest/api_guide/9c4dec/create-trainingsets/) in Domino. This is the baseline for data drift detection.

(3) Spin up a [Domino Model API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/deploy-models-at-rest/) from the registered model.

(4) Once your Domino Model API is running, register your model with Domino Model Monitoring from the Model API UI.

### II. Capture Data Drift

(5) Send some test data to your model. Domino will automatically capture this scoring data and the corresponding predictions for you. Wait until the initial inference data has ingested (this takes about an hour the first time)

(6) Schedule drift data checks in Domino Model Monitoring.

### III. (Optional) Capture ground truth labels for Model Quality monitoring

(7) Register an external datasource in both Domino Model Monitoring and in the Workbench.

(8) Upload a dataset with ground truth labels from the Workbench to the external datasource.

(9) Provide Domino Model monitoring the path to your ground truth data file(s) (either in the UI or via API). 

(10) Schedule ground truth monitoring checks in Domino Model Monitoring.


## Custom Metric Example

Domino Model monitoring can also capture alerts and generate notifications for custom metrics.

**4_Custom_Metrics_Quickstart.ipynb** walks through creating a custom metric & notifications.
