'''
Light Gradient Boosting Model function that is used to generate trials with different hyperparameter setups and determine the best model

using Gradient Boosted Descision Trees
'''

# %% --------------------------------------------------------------------------
# 1: Imports
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay,classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import argparse
import mlflow
import optuna
import logging


# %% --------------------------------------------------------------------------
# 2: Setup Logging and MLFlow
# -----------------------------------------------------------------------------

#MLFlow was used as a tracking tool to allow us to run lots of models and have a clear way to track them
logging.basicConfig(format='[%(levelname)s %(module)s] %(asctime)s - %(message)s',level = logging.INFO)

logger = logging.getLogger(__name__)

experiment_name = 'bank_churn_projectV2'
parser = argparse.ArgumentParser(description = experiment_name)

tracking_uri = r'file:///C:/mlflow_local/mlruns'
logger.info(f'Tracking uri set to {tracking_uri}')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)


# %% --------------------------------------------------------------------------
# 3: Import Data
# -----------------------------------------------------------------------------

df = pd.read_csv('monthly_churn_with_econ.csv',index_col=0)
df.drop(columns=['DATE','creation_date','customer_id'],inplace=True)
# %% --------------------------------------------------------------------------
# 4: Dataset Preperation
# -----------------------------------------------------------------------------

rng = np.random.RandomState(1234)
X = df.drop(columns=['churn'])
#X['state'] = LabelEncoder().fit_transform(X['state'])  
y = df['churn']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)

metric = 'f1'
dic = {'cols' : list(X.columns),'hp_tuning_metric':metric}

# %% --------------------------------------------------------------------------
# 3: Hyperparameter Experiment Function
# -----------------------------------------------------------------------------

def objective(trial):

    list_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250,500,750,1000] 

    #Setting the hyperparameter ranges available during tuning (essentially binding the search space for the model)
    params = {
        'n_estimators': trial.suggest_int('n_estimators',50,1000),              #Number of boosting iterations
        'max_depth': trial.suggest_int('max_depth',1,10),                       #Max depth per tree
        'learning_rate': trial.suggest_float('learning_rate',0.001,0.5),        #Rate to adjust feature weights in response to the error residual
        'subsample': trial.suggest_float('subsample',0.5,1.0),                  #What ratio of the total dataset is being used to sample for each boostrapped sample
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),         #L1 Norm coeff
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),       #L2 Norm coeff
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3,1),     #What fraction of total features to be considered for each tree in the model
        'num_leaves' : trial.suggest_int('num_leaves', 2, 1000),                #Max number of leaves in each tree
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 400),    #Minimum number of samples required in the child node in order to create it
        'max_bin': trial.suggest_categorical('max_bin', list_bins),             #max number of bins that feature values will be bucketed in
        'scale_pos_weight': trial.suggest_int('scale_pos_weight',10,5000)       #Scales the weighting of the positive class, used to handle unbalanced datasets
        }

    #Start Hyperparameter tuning with mlflow logging
    with mlflow.start_run(nested = True):
        mlflow.log_params(params)
        mlflow.log_dict(params,'params.json')

        #Defining the LGBM Model
        classifier_obj = lgb.LGBMClassifier(**params,
                                                random_state=1234,
                                                boosting_type='gbdt',       #Gradient Boosted Descision Tree submodels
                                                verbose = -1                #Supress the information reporting output

        )

        # Generating the cross-validation sets with random sampling and stratification
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=rng)
        score = cross_val_score(classifier_obj,X_train,y_train, n_jobs=-1, cv=cv,scoring=metric)
        accuracy = abs(score.mean())

        #logging our f1-score to MLFlow
        mlflow.log_metric('f1_val',accuracy)
        
        #Fitting and predicting the model for this given set of hyperparameters
        classifier_obj.fit(X_train,y_train)
        y_pred_X_train = classifier_obj.predict(X_train)
        y_pred_X_test = classifier_obj.predict(X_test)
        y_pred_p = classifier_obj.predict_proba(X_test)[:, 1]

        #Logging all the relevent metrics to the MLFlow dashboard
        mlflow.log_metric('roc_train',roc_auc_score(y_train,y_pred_X_train))
        mlflow.log_metric('roc_test',roc_auc_score(y_test,y_pred_X_test))

        mlflow.log_metric('accuracy_train',accuracy_score(y_train,y_pred_X_train))
        mlflow.log_metric('accuracy_test',accuracy_score(y_test,y_pred_X_test))

        mlflow.log_metric('f1_score_train',f1_score(y_train,y_pred_X_train))
        mlflow.log_metric('f1_score_test',f1_score(y_test,y_pred_X_test))

        mlflow.log_metric('precision_train',precision_score(y_train,y_pred_X_train))
        mlflow.log_metric('precision_test',precision_score(y_test,y_pred_X_test))

        mlflow.log_metric('recall_train',recall_score(y_train,y_pred_X_train))
        mlflow.log_metric('recall_test',recall_score(y_test,y_pred_X_test))

        #Plotting a confusion matrix and submitting it to the MLFlow dashoard as a png
        fig,ax = plt.subplots()
        cm = confusion_matrix(y_test,y_pred_X_test)
        ConfusionMatrixDisplay(cm).plot(ax=ax,cmap = 'OrRd')
        mlflow.log_figure(fig, 'confusion_matrix.png')

        #Plotting RoC and Precision-recall curve and logging them to MLFlow also
        fpr, tpr, _ = roc_curve(y_test, y_pred_p)
        prec, reca, _ = precision_recall_curve(y_test, y_pred_p)

        fig1 = plt.figure()
        plt.plot(fpr,tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        mlflow.log_figure(fig1, 'RocCurve.png')

        fig2 = plt.figure()
        plt.plot(prec,reca)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        mlflow.log_figure(fig2, 'PRCurve.png')

        mlflow.lightgbm.log_model(classifier_obj,'model')
    return accuracy

#if run as a script, create a new optuna study using the objective function and run 50 trial
if __name__ == '__main__':
    with mlflow.start_run():
        sampler = optuna.samplers.TPESampler(seed=1234)
        study = optuna.create_study(direction='maximize',sampler=sampler)
        study.optimize(objective, n_trials=50)

    print(study.best_trial)
    print(study.best_params)
    
    fig1 = optuna.visualization.plot_param_importances(study)
    plt.savefig('figures/xgb_param_importance.png')
    fig1.show()

    fig2 = optuna.visualization.plot_optimization_history(study)
    plt.savefig('figures/xgb_optimization_history.png')
    fig2.show()