#%%
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay,classification_report, roc_auc_score, roc_curve, precision_recall_curve, make_scorer,fbeta_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from joblib import dump


#%%

df = pd.read_csv('../data/Modelling_DATA.csv',index_col=0)
#%%
df['index'] = df.index
last = df.groupby('account_id').last()

#%%
df.drop(index = last['index'],inplace=True)
df.drop(columns = ['account_id','index'],inplace=True)

#%%

rng = np.random.RandomState(1234)
X = df.drop(columns=['CHURN','churn0.5'])
# X['state'] = LabelEncoder().fit_transform(X['state']) 
y = df['CHURN']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)

col_trans = ColumnTransformer([
    ('scale',MinMaxScaler(),X.columns)
])

Xt_train = col_trans.fit_transform(X_train)
Xt_test = col_trans.transform(X_test)

dump(col_trans,'../app/resources/model/transformerV3.joblib')

Xt_train = pd.DataFrame(Xt_train, columns=col_trans.get_feature_names_out())
Xt_test = pd.DataFrame(Xt_test, columns=col_trans.get_feature_names_out())





#%%

lgb_params = {
  "n_estimators": 894,
  "max_depth": 9,
  "learning_rate": 0.4995228200231161,
  "subsample": 0.6222855611176402,
  "reg_alpha": 0.005419620478683748,
  "reg_lambda": 0.40121771253043403,
  "colsample_bytree": 0.5108923470662523,
  "num_leaves": 290,
  "min_child_samples": 74,
  "max_bin": 175,
  "scale_pos_weight": 3866
}

model = lgb.LGBMClassifier(**lgb_params,
                            random_state=1234,
                            boosting_type='gbdt',
                            verbose = -1
        )

model.fit(Xt_train,y_train)
#%%
y_pred = model.predict(Xt_test)
#%%

dump(model,'../app/resources/model/modelV3.joblib')

#%%
print("Accuracy = {}".format(accuracy_score(y_test,y_pred)))
print("F1 = {}".format(f1_score(y_test,y_pred)))
print('Recall = {}'.format(recall_score(y_test,y_pred)))
print('Precision = {}'.format(precision_score(y_test,y_pred)))

#%%

fig,ax = plt.subplots()
cm = confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot(ax=ax,cmap = 'OrRd')
plt.show()

#%%


