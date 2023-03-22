# Banking_Churn_Main
Banking Churn Project repo created by David Neilan

## Introduction
This repo includes all files related to the project except for some of the datasets that were too large to push to github. Additionally there is a pdf file of the powerpoint presentation we made to explain our workflow called 'Banking Churn Project.pdf'

## Folder Breakdown
### app
 The app folder contains all files relevent to the flask app that can be containerized to act as a predictive service for this model. This includes the model files pipelines and data cleaning. (The general pipeline for the service flows as: example_request.py -> main.py -> Classifier.py -> Data_generation.py -> Time_Series.py (Optional) )

### archived_code 
Contains all legacy code no longer relevent to the model or the service itself.

### data 
Contains *some* of the datasets used in the project though some were too large to publish here

### figures
Stores some of the figures used during EDA, feature engineering and feature importance steps

### model selection
Stores all the files used to train and hyperparameter tune the sub models relevent to the stacked ensemble meta-model (lgb_model_selection.py is the most documented)
