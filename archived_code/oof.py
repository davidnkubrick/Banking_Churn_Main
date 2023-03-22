#%%
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

#%%

df = pd.read_csv('data/monthly_churn_with_econ.csv',index_col=0)
#%%

rng = np.random.RandomState(1234)
X = df.drop(columns=['CHURN'])
y = df['CHURN']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)
#%%

import warnings
warnings.filterwarnings('ignore')

n_folds = 10
cv = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=rng)

lgb_train_preds = np.zeros(len(X_train.index))
lgb_test_preds = np.zeros(len(X_test.index))

target = y_train

for folds, (train_index,test_index) in enumerate(cv.split(X_train,target)):
    print(f'FOLD {folds+1}')

    y_train = target.iloc[train_index]
    y_valid = target.iloc[test_index]

    lgb_x_train = X_train.iloc[train_index]
    lgb_x_valid = X_train.iloc[test_index]

    lgb_params = {
                "n_estimators": 668,
                "max_depth": 9,
                "learning_rate": 0.05877265398892562,
                "subsample": 0.6612013241074571,
                "reg_alpha": 0.0000025842841164395406,
                "reg_lambda": 0.403282066083172,
                "colsample_bytree": 0.43228071674346247,
                "num_leaves": 772,
                "min_child_samples": 233,
                "max_bin": 175,
                "scale_pos_weight": 3898
}
    
    lgb_model = lgb.LGBMClassifier(**lgb_params,random_state=123)
    lgb_model.fit(
        lgb_x_train,
        y_train,
        verbose = -1
    )

    train_oof_preds = lgb_model.predict_proba(lgb_x_valid)[:,1]
    test_oof_preds = lgb_model.predict_proba(X_test)[:,1]
    lgb_train_preds[test_index] = train_oof_preds
    lgb_test_preds += test_oof_preds /n_folds

    lgb

    print("LGB - F1 = {}".format(f1_score(y_valid, train_oof_preds.round(0))))
    print('LGB - Recall = {}'.format(recall_score(y_valid,train_oof_preds.round())))
    print('LGB Precision = {}'.format(precision_score(y_valid,train_oof_preds.round(0))))
    print('')

    # if roc_auc_score(y_valid, train_oof_preds) > threshold:
    #     test_oof_preds = lgb_model.predict_proba(X_test)[:,1]
    #     lgb_test_preds += test_oof_preds
    #     count1 += 1

#%%
print("--> Overall train metrics")
print("LGB - F1 = {}".format(f1_score(target, lgb_train_preds.round(0))))
print('LGB - Recall = {}'.format(recall_score(target,lgb_train_preds.round(0))))
print('LGB Precision = {}'.format(precision_score(target,lgb_train_preds.round(0))))

print("--> Overall test metrics")
print(": LGB - F1 = {}".format(f1_score(y_test,lgb_test_preds.round(0))))
print('LGB - Recall = {}'.format(recall_score(y_test,lgb_test_preds.round(0))))
print('LGB Precision = {}'.format(precision_score(y_test,lgb_test_preds.round(0))))

#%%

fig,ax = plt.subplots()
cm = confusion_matrix(y_test,lgb_test_preds.round(0))
ConfusionMatrixDisplay(cm).plot(ax=ax,cmap = 'OrRd')
plt.show()