#%%
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay,classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import catboost as cat
import argparse
import mlflow
import optuna
import logging

#%%

logging.basicConfig(format='[%(levelname)s %(module)s] %(asctime)s - %(message)s',level = logging.INFO)

logger = logging.getLogger(__name__)

experiment_name = 'bank_churn_project_lgb'
parser = argparse.ArgumentParser(description = experiment_name)

tracking_uri = r'file:///C:/mlflow_local/mlruns'
logger.info(f'Tracking uri set to {tracking_uri}')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)


#%%
file_name = 'Modelling_DATA.csv' # ENTER CSV FILE NAME  HERE, INCLUDING .csv
df = pd.read_csv('../data/' + file_name,index_col=0)
#%%

rng = np.random.RandomState(1234)
X = df.drop(columns=['CHURN'])
# X['state'] = LabelEncoder().fit_transform(X['state']) 
y = df['CHURN']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)

#%%

metric = 'f1'
dic = {'cols' : list(X.columns),'hp_tuning_metric':metric}

#%%

def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators',50,1000),
        'max_depth': trial.suggest_int('max_depth',1,10),
        'learning_rate': trial.suggest_float('learning_rate',0.001,0.5),
        'subsample': trial.suggest_float('subsample',0.5,1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight',1,5000)
        }

    with mlflow.start_run(nested = True):
        mlflow.log_params(params)
        mlflow.log_dict(params,'params.json')

        classifier_obj = cat.CatBoostClassifier(**params,
                                                random_state=123,
                                                verbose=False
        )

        cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=rng)
        score = cross_val_score(classifier_obj,X_train,y_train, n_jobs=-1, cv=cv,scoring=metric)
        accuracy = abs(score.mean())
        # classifier_obj.fit(X_train,y_train)
        # y_pred = classifier_obj.predict(X_test)
        # accuracy = f1_score(y_test,y_pred)

        mlflow.log_metric('f1_val',accuracy)
            
        classifier_obj.fit(X_train,y_train)
        y_pred_X_train = classifier_obj.predict(X_train)
        y_pred_X_test = classifier_obj.predict(X_test)
        y_pred_p = classifier_obj.predict_proba(X_test)[:, 1]


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


        fig,ax = plt.subplots()
        cm = confusion_matrix(y_test,y_pred_X_test)
        ConfusionMatrixDisplay(cm).plot(ax=ax,cmap = 'OrRd')
        mlflow.log_figure(fig, 'confusion_matrix.png')


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

        mlflow.catboost.log_model(classifier_obj,'model')
    return accuracy


if __name__ == '__main__':
    with mlflow.start_run():
        sampler = optuna.samplers.TPESampler(seed=1234)
        study = optuna.create_study(direction='maximize',sampler=sampler)
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=15)

    print(study.best_trial)
    print(study.best_params)
    
    fig1 = optuna.visualization.plot_param_importances(study)
    plt.savefig('figures/lgb_param_importance.png')
    fig1.show()

    fig2 = optuna.visualization.plot_optimization_history(study)
    plt.savefig('figures/lgb_optimization_history.png')
    fig2.show()