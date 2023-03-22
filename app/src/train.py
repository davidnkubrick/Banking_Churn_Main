#%%
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cat
from sklearn.ensemble import StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from joblib import dump

#%%
df = pd.read_csv('../../data/Modelling_DATA.csv',index_col=0)
#%%
df['index'] = df.index
last = df.groupby('account_id').last()

#%%
df.drop(index = last['index'],inplace=True)
df.drop(columns = ['account_id','index'],inplace=True)
#%%

rng = np.random.RandomState(123)
X = df.drop(columns=['CHURN'])
y = df['CHURN']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)

col_trans = ColumnTransformer([
    ('scale',MinMaxScaler(),X.columns)
])

Xt_train = col_trans.fit_transform(X_train)
Xt_test = col_trans.transform(X_test)

dump(col_trans,'../resources/model/transformerV6.joblib')

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

sgd_params = {
  "alpha": 0.00005851775845135216,
  "l1_ratio": 0.4909307939002558,
  "max_iter": 1218
}

cat_params = {
  "n_estimators": 699,
  "max_depth": 8,
  "learning_rate": 0.18575512664040708,
  "subsample": 0.7805980930328125,
  "min_data_in_leaf": 11,
  "scale_pos_weight": 69
}

rf_params = {'n_estimators': 50,
             'max_depth': 10}


lgb_model = lgb.LGBMClassifier(**lgb_params,
                               random_state=123,
                                boosting_type='gbdt',
                                verbose = -1)

sgd_model = CalibratedClassifierCV(
    SGDClassifier(**sgd_params,
        loss='log_loss',
        random_state=123,
        penalty='elasticnet'
    )
)

cat_model = cat.CatBoostClassifier(**cat_params,
                                random_state=123,
                                verbose=False
                                   )

rf_model = RandomForestClassifier(**rf_params,
                                random_state = 123
                                  )

base_models = [('lgb',lgb_model),
                ('sgd',sgd_model),
                ('cat',cat_model),
                ('rf',rf_model)
                ]

cv = StratifiedKFold(n_splits = 5,shuffle = True, random_state = rng)
stack = StackingClassifier(estimators = base_models, cv = cv)

stack.fit(Xt_train,y_train)
y_pred = stack.predict(Xt_test)

#%%
dump(stack,'../resources/model/modelV6.joblib')

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