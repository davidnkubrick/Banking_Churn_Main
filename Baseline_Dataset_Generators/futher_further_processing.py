#%%
import pandas as pd
import numpy as np
import datetime as dt
#%%

df = pd.read_csv('../data/new_churn2.csv',index_col=0)

#%%
grouped = df.groupby('customer_id')

df['CHURN'] = grouped['churn2'].shift(periods=-1,axis=0)
df['CHURN'] = df['CHURN'].fillna(0)
#%%

df.to_csv('../data/new_churn_shap.csv',index = False)
#%%