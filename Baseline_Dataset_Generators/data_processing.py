#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

#%%
cust = pd.read_csv('../raw_data/customers_tm1_e.csv')
trans = pd.read_csv('../raw_data/transactions_tm1_e.csv')

cust.drop(index = cust[cust.start_balance>1000000].index,inplace=True)

df = pd.merge(left = cust,right = trans,on = 'customer_id',how = 'right')


#%%

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['date'] = pd.to_datetime(df['date'])

df['age'] = (df['transaction_date'] - pd.to_datetime(df['dob'])).astype('timedelta64[Y]')
# df.drop(columns='dob',inplace=True)
#%%

grouped = df.groupby('account_id')
sum_dep = grouped['deposit'].transform('sum')
sum_with = grouped['withdrawal'].transform('sum')

df['end_balance'] = df.start_balance + sum_dep + sum_with 
df['last_transaction_date'] = grouped['transaction_date'].transform(max)
df['last_transaction'] = df.apply(lambda x: 1 if x['date'] == x['last_transaction_date'] else 0,axis=1)
#%%

df['account_closed'] = df.apply(lambda x: 1 if (x['last_transaction']==1) & (x['end_balance'] == 0) else 0,axis=1)
df['closure_date'] = df[df.account_closed ==1 ].date

#%%
def following_month(date,closure_date):
    if type(closure_date) == pd.Timestamp:
        if (date.month == 12) and (closure_date.month == 1) and (closure_date.year == (date.year +1)):
            return True
        elif (closure_date.month == (date.month + 1)) & (closure_date.year == date.year):
            return True
        else:
            return False
    else:
        return False

df['closure_date'] = grouped['closure_date'].transform(max)
df['churn'] = df.apply(lambda x: 1 if following_month(x.date,x.closure_date) == True else 0,axis=1)

df['cum_sum_deposit'] = grouped['deposit'].cumsum()
df['cum_sum_withdraw'] = grouped['withdrawal'].cumsum()
df['cum_net'] = df.cum_sum_withdraw + df.cum_sum_deposit

df['account_age'] = (df['date'] - df['creation_date'])
df['account_age'] = df['account_age'].apply(lambda x: x.days)
df['balance'] = df['start_balance'] + df['cum_sum_withdraw'] + df['cum_sum_deposit']
df['transaction_number'] = grouped.cumcount()
#%%

df['state'] = df['state'].apply(lambda x: x.title() if type(x) == str else x)
df['state'] = df['state'].apply(lambda x: 'Texas' if x == 'Tx' else x)
df['state'] = df['state'].apply(lambda x: 'Massachusetts' if x == 'Mass' else x)
df['state'] = df['state'].apply(lambda x: 'New York' if x == 'Ny' else x)
df['state'] = df['state'].apply(lambda x: 'Florida, USA' if x == 'Florida' else x)
df.drop(index = df[df.state == '-999'].index,inplace=True)
df.drop(index = df[df.state == 'Australia'].index,inplace=True)
df.drop(index = df[df.state == 'Unk'].index,inplace = True)

df.drop(index = df[df.start_balance<0].index,inplace=True)

df['balance'] = df['balance'].apply(lambda x: 0 if x<0.01 else x)

# df['balance'] = df['balance'].apply()

df.drop(columns=['last_transaction','closure_date','account_id','end_balance'],inplace=True)
#%%
# df.drop(columns=['last_transaction','closure_date','account_id','creation_date','last_transaction_date','account_closed','end_balance','date','transaction_date','customer_id'],inplace=True)
df.dropna(axis = 'index',inplace=True)
df.to_csv('../data/processed_data_raw.csv')
# df.to_csv('processed_data.csv')
#%%
