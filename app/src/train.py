'''
Train.py

Train the 4 models that make up the stacked ensemble log-reg metamodel then train the metamodel itself
'''


# %% --------------------------------------------------------------------------
# 1: Imports
# -----------------------------------------------------------------------------

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


# %% --------------------------------------------------------------------------
# 2: Import data
# -----------------------------------------------------------------------------

df = pd.read_csv('../../data/Modelling_DATA.csv',index_col=0)

# %% --------------------------------------------------------------------------
# 3: Final Data Cleaning
# -----------------------------------------------------------------------------

df['index'] = df.index
last = df.groupby('account_id').last() #stripping the final row for each customer as the data cannot be labelled
df.drop(index = last['index'],inplace=True) # drop the final row so that we are left with the modelling dataset
df.drop(columns = ['account_id','index'],inplace=True) #removing non-modelling features


# %% --------------------------------------------------------------------------
# 4: Modelling Preperations
# -----------------------------------------------------------------------------

rng = np.random.RandomState(123) #Random state set to ensure repeatability

#Splitting data into features and label dataframes X, y 
X = df.drop(columns=['CHURN'])
y = df['CHURN']

#Splitting the dataset into test and train sets, stratified to ensure no bias occurs through class imbalance (only 20% of dataset is labelled 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=rng,stratify = y)


#Scaling all numerical columns using a minmax as there is a large difference in scale between some features
col_trans = ColumnTransformer([
    ('scale',MinMaxScaler(),X.columns)
])

#Fit the transformer using the training set and then transform both the train and test to avoid data leakage
Xt_train = col_trans.fit_transform(X_train)
Xt_test = col_trans.transform(X_test)

dump(col_trans,'../resources/model/transformerV6.joblib')

Xt_train = pd.DataFrame(Xt_train, columns=col_trans.get_feature_names_out())
Xt_test = pd.DataFrame(Xt_test, columns=col_trans.get_feature_names_out())

# %% --------------------------------------------------------------------------
# 5: Sub-model Definitions
# -----------------------------------------------------------------------------
'''
All models were run through iterations of hyperparmeter tuning using MLFlow and
OPTuna to arrive at the hyperparameter values shown bellow.
'''
#LightGBM: gradient boosting model.  F1 = 0.495, Recall = 0.938, Precision = 0.336
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

#SGD: Stochastic gradient decent model using logistic regression elastic net penalty term. F1 = 0.37, Recall = 0.285, Precision = 0.527
sgd_params = {
  "alpha": 0.00005851775845135216,
  "l1_ratio": 0.4909307939002558,
  "max_iter": 1218
}

#CatBoost: Gradient boosting model designed for categorical features. F1 = 0.456, Recall = 0.903, Precision = 0.297
cat_params = {
  "n_estimators": 699,
  "max_depth": 8,
  "learning_rate": 0.18575512664040708,
  "subsample": 0.7805980930328125,
  "min_data_in_leaf": 11,
  "scale_pos_weight": 69
}

#Random Forrest: Parellelised ensemble of descision trees. F1 = 0.423, Recall = 0.322, Precision = 0.616
rf_params = {'n_estimators': 50,
             'max_depth': 10}

# %% --------------------------------------------------------------------------
# 6: Sub-model Construction
# -----------------------------------------------------------------------------
lgb_model = lgb.LGBMClassifier(**lgb_params,
                               random_state=123,
                                boosting_type='gbdt',   # Gradient boosted decision trees
                                verbose = -1)           # Suppressing reporting outputs during model training

sgd_model = CalibratedClassifierCV(                     # Calibrating the output probabilities of the SGDClassifier to be between 0 and 1 by passing them through a sigmoid
    SGDClassifier(**sgd_params,                         
        loss='log_loss',                                # Using a log_loss function makes the model a logistic regression probabilistic classifier
        random_state=123,
        penalty='elasticnet'                            # Elastic net to bring some term of sparsity (feature reduction) to the model that would not be possible with an L2 penalty
    )
)

cat_model = cat.CatBoostClassifier(**cat_params,
                                random_state=123,
                                verbose=False
                                   )

rf_model = RandomForestClassifier(**rf_params,
                                random_state = 123
                                  )


# %% --------------------------------------------------------------------------
# 7: Meta-model Construction
# -----------------------------------------------------------------------------
base_models = [('lgb',lgb_model),
                ('sgd',sgd_model),
                ('cat',cat_model),
                ('rf',rf_model)
                ]

cv = StratifiedKFold(n_splits = 5,shuffle = True, random_state = rng) 
stack = StackingClassifier(estimators = base_models, cv = cv)


# %% --------------------------------------------------------------------------
# 8: Meta-model Fitting and Prediction
# -----------------------------------------------------------------------------
stack.fit(Xt_train,y_train)
y_pred = stack.predict(Xt_test)

dump(stack,'../resources/model/modelV6.joblib')

# %% --------------------------------------------------------------------------
# 9: Meta-model Reporting
# -----------------------------------------------------------------------------
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