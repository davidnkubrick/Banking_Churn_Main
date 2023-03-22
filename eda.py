#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import shap
import xgboost as xgb
import lightgbm as lgb
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler



#%%

df = pd.read_csv('data/Modelling_DATA.csv',index_col=0)
#%%
# fig,ax = plt.subplots()
# g = sns.heatmap(abs(df.corr()),cmap = 'YlGnBu',ax=ax)
# plt.gcf().set_size_inches(20,20)
# plt.savefig('figures/heatmap_corr.png')

# ncols = 2
# nrows = np.ceil(len(df.columns)/ncols).astype(int)
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,nrows*2.5))
# for c, ax in zip(df.columns, axs.flatten()):
#     sns.histplot(df, x=c, ax=ax)
# fig.suptitle('Distribution of all variables', fontsize=20)
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.savefig('figures/data_distribution.png')
# plt.show()

#%%

rng = np.random.RandomState(123)
X = df.drop(columns=['CHURN'])
y = df['CHURN']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=rng,stratify = y)

col_trans = ColumnTransformer([
    ('scale',MinMaxScaler(),X.columns)
])

# dump(col_trans,'../resources/transformer.joblib')

Xt_train = col_trans.fit_transform(X_train)
Xt_test = col_trans.transform(X_test)



Xt_train = pd.DataFrame(Xt_train, columns=col_trans.get_feature_names_out())
Xt_test = pd.DataFrame(Xt_test, columns=col_trans.get_feature_names_out())

#%%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
sgd_params = {
  "alpha": 0.00005851775845135216,
  "l1_ratio": 0.4909307939002558,
  "max_iter": 1218
}

model = CalibratedClassifierCV(
    SGDClassifier(**sgd_params,
        loss='log_loss',
        random_state=123,
        penalty='elasticnet'
    )
)

model.fit(Xt_train,y_train)

#%%
explainer_ebm = shap.Explainer(model.predict,Xt_train)
# explainer_ebm = shap.TreeExplainer(model)
shap_values_ebm = explainer_ebm(Xt_train)

#%%
shap.plots.beeswarm(shap_values_ebm,show = False,plot_size=(10,15),max_display=30)
plt.savefig('figures/shap_sgd_V1/shap_beeswarm.png',dpi = 300,bbox_inches ='tight')
plt.close()

shap.plots.waterfall(shap_values_ebm[0],max_display=30,show=False)
plt.gcf().set_size_inches(15,10)
plt.savefig('figures/shap_sgd_V1/shap_waterfall.png')
plt.close()

shap.plots.bar(shap_values_ebm,max_display=30,show=False)
plt.gcf().set_size_inches(20,20)
plt.savefig('figures/shap_sgd_V1/shap_bar.png')
plt.close()

#%%
