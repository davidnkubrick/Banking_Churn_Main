
#%%
import pandas as pd
import numpy as np
import datetime as dt

#%%

df = pd.read_csv('../data/processed_data_raw.csv',index_col=0)

df['date'] = pd.to_datetime(df['date'])
#%%

df['month_year'] = df['date'].dt.to_period('M')

group_month = df.groupby(['customer_id', 'month_year'])

df['monthly_withdrawal'] = group_month['withdrawal'].transform(sum)
df['monthly_deposit'] = group_month['deposit'].transform(sum)
df['monthly_net'] = df['monthly_deposit'] + df['monthly_withdrawal']

df['zero_amount'] = df['amount'].apply(lambda x: 1 if abs(x) < 0.03 else 0)
df['non_zero_amount'] = df['amount'].apply(lambda x: 1 if abs(x) >= 0.03 else 0)
df['withdrawal_occurs'] = df['withdrawal'].apply(lambda x: 1 if abs(x) != 0 else 0)
df['deposit_occurs'] = df['deposit'].apply(lambda x: 1 if abs(x) != 0 else 0)

df['monthly_interactions'] = group_month['zero_amount'].transform(sum)
df['monthly_transactions'] = group_month['non_zero_amount'].transform(sum)
df['no_of_monthly_withdrawals'] = group_month['withdrawal_occurs'].transform(sum)
df['no_of_monthly_deposits'] = group_month['deposit_occurs'].transform(sum)


#%%
df.drop(columns = ['zero_amount','non_zero_amount','withdrawal_occurs','deposit_occurs'],inplace=True)

#%%
features = ['customer_id','creation_date','start_balance','state','month_year','monthly_withdrawal','monthly_deposit','monthly_interactions','monthly_transactions','no_of_monthly_withdrawals','no_of_monthly_deposits','dob','churn']
df_month = df[features]
df_month.drop_duplicates(inplace=True)

#%%

grouped = df_month.groupby('customer_id')
df_month['total_cum_sum_withdrawal'] = grouped['monthly_withdrawal'].cumsum()
df_month['total_cum_sum_deposit'] = grouped['monthly_deposit'].cumsum()
df_month['end_of_month_balance'] = df_month.start_balance + df_month.total_cum_sum_withdrawal + df_month.total_cum_sum_deposit

df_month['total_cum_count_transactions'] = grouped['monthly_transactions'].cumcount()
df_month['total_cum_count_interactions'] = grouped['monthly_interactions'].cumcount()

#%%

df_month['creation_date'] = pd.to_datetime(df_month['creation_date'])
df_month['months_from_creation'] = (df_month.month_year.view(dtype='int64') - df_month['creation_date'].dt.to_period('M').view(dtype='int64'))#.astype(np.timedelta64('M'))

#%%

# # df_month['no_of_deposits_last_3_months'] = df.agg()
# df_month.set_index('Date',inplace=True)
# df_month['no_of_deposits_last_3_months'] = df_month.resample('3M')['no_of_monthly_withdrawals'].count()
# df.reset_index(inplace=True)
# df_month.drop(columns=['cumsum_with','cumsum_dep'],inplace=True)

#%%

# df_month.to_csv('monthly_churn.csv',index = False)

#%%

macro = pd.read_csv('../data/macroecon_features.csv',index_col=0)
macro['DATE'] = pd.to_datetime(macro.DATE)
macro['DATE'] = macro['DATE'].dt.to_period('M')

churn_dat = df_month

# %% --------------------------------------------------------------------------# 
# making sure they look ok
# ----------------------------------------------------------------------------- 
print(macro)
print(churn_dat)
# %% --------------------------------------------------------------------------
# merging them# -----------------------------------------------------------------------------
df_econ = pd.merge(macro,churn_dat,left_on='DATE',right_on='month_year',how = 'right').drop('month_year',axis = 1)

#%%
df_econ.to_csv('../data/monthly_churn_with_econV2.csv')

#%%