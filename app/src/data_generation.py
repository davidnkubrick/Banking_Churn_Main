#%%
import pandas as pd
import numpy as np
from datetime import date


def make_test_df(samp):
    cust = pd.read_csv('../resources/data/customers_tm1_e.csv')
    trans = pd.read_csv('../resources/data/transactions_tm1_e.csv')

    samp['account_id'] = samp.account_id.astype(int)

    new_trans = trans[trans.account_id.isin(samp.account_id)]

    customer_id = new_trans.customer_id

    new_cust = cust[cust.customer_id.isin(customer_id)]
    
    del cust
    del trans

    return new_cust,new_trans

def initial_processing(cust,trans):

    # Allow a df or file_path to be passed into the function
    if type(cust) == str:
        cust = pd.read_csv(cust)
    if type(trans) == str:
        trans = pd.read_csv(trans)

    #Taken from json so make sure all numerical columns are in the right format
    cust['start_balance'] = cust['start_balance'].astype(float)
    cust['customer_id'] = cust['customer_id'].astype(int)
    trans['account_id'] = trans['account_id'].astype(int)
    trans['customer_id'] = trans['customer_id'].astype(int)
    trans['amount'] = trans['amount'].astype(float).round(2)
    trans['deposit'] = trans['deposit'].astype(float).round(2)
    trans['withdrawal'] = trans['withdrawal'].astype(float).round(2)


    # Drop outliers from start balance - 2 samples with balance above a billion and 1 sample with negative balance
    cust.drop(index = cust[cust['start_balance']>1000000].index,inplace=True)
    cust.drop(index = cust[cust['start_balance']<0].index,inplace=True)

    # Merge the datasets
    df = pd.merge(left = cust,right = trans,on = 'customer_id',how = 'right')

    # Convert date strings to date time
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    df['date'] = pd.to_datetime(df['date'])

    grouped = df.groupby('account_id')
    sum_dep = grouped['deposit'].transform('sum')
    sum_with = grouped['withdrawal'].transform('sum')

    # Used to calculate initial definition of churn, not relevant for the final revenue churn definition
    df['end_balance'] = df.start_balance + sum_dep + sum_with 
    df['last_transaction_date'] = grouped['transaction_date'].transform(max)
    df['last_transaction'] = df.apply(lambda x: 1 if x['date'] == x['last_transaction_date'] else 0,axis=1)

    df['account_closed'] = df.apply(lambda x: 1 if (x['last_transaction']==1) & (x['end_balance'] == 0) else 0,axis=1)
    df['closure_date'] = df[df.account_closed ==1 ].date

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

    # General feature engineering
    df['cum_sum_deposit'] = grouped['deposit'].cumsum()
    df['cum_sum_withdraw'] = grouped['withdrawal'].cumsum()
    df['cum_net'] = df.cum_sum_withdraw + df.cum_sum_deposit

    df['account_age'] = (df['date'] - df['creation_date'])
    df['account_age'] = df['account_age'].apply(lambda x: x.days)
    df['balance'] = df['start_balance'] + df['cum_sum_withdraw'] + df['cum_sum_deposit']
    df['transaction_number'] = grouped.cumcount()

    # Clean up issues with the state column
    df['state'] = df['state'].apply(lambda x: x.title() if type(x) == str else x)
    df['state'] = df['state'].apply(lambda x: 'Texas' if x == 'Tx' else x)
    df['state'] = df['state'].apply(lambda x: 'Massachusetts' if x == 'Mass' else x)
    df['state'] = df['state'].apply(lambda x: 'New York' if x == 'Ny' else x)
    df.drop(index = df[df.state == '-999'].index,inplace=True)
    df.drop(index = df[df.state == 'Australia'].index,inplace=True)
    df.drop(index = df[df.state == 'Unk'].index,inplace = True)

    df['balance'] = df['balance'].apply(lambda x: 0 if x<0.01 else x)

    # Drop the columns used to calculate the old definition of churn and customer_id
    df.drop(columns=['last_transaction','closure_date','customer_id','end_balance'],inplace=True)

    # Some nan columns from missing info in customer database (5 customers with missing info), drop these
    df.dropna(axis = 'index',inplace=True)
    return df

def monthly_data(df,econ_data_file_name):
    
    df['month_year'] = df['date'].dt.to_period('M')

    group_month = df.groupby(['account_id', 'month_year'])

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

    df.drop(columns = ['zero_amount','non_zero_amount','withdrawal_occurs','deposit_occurs'],inplace=True)

    features = ['account_id','creation_date','start_balance','state','month_year','monthly_withdrawal','monthly_deposit','monthly_interactions','monthly_transactions','no_of_monthly_withdrawals','no_of_monthly_deposits','dob','churn']
    df_month = df[features]
    df_month.drop_duplicates(inplace=True)

    grouped = df_month.groupby('account_id')
    df_month['total_cum_sum_withdrawal'] = grouped['monthly_withdrawal'].cumsum()
    df_month['total_cum_sum_deposit'] = grouped['monthly_deposit'].cumsum()
    df_month['end_of_month_balance'] = df_month.start_balance + df_month.total_cum_sum_withdrawal + df_month.total_cum_sum_deposit

    df_month['total_cum_count_transactions'] = grouped['monthly_transactions'].cumcount()
    df_month['total_cum_count_interactions'] = grouped['monthly_interactions'].cumcount()

    # df_month['creation_date'] = pd.to_datetime(df_month['creation_date'])
    # df_month['months_from_creation'] = (df_month.month_year.view(dtype='int64') - df_month['creation_date'].dt.to_period('M').view(dtype='int64'))#.astype(np.timedelta64('M'))

    macro = pd.read_csv(econ_data_file_name,index_col=0)
    macro['DATE'] = pd.to_datetime(macro.DATE)
    macro['DATE'] = macro['DATE'].dt.to_period('M')

    df_econ = pd.merge(macro,df_month,left_on='DATE',right_on='month_year',how = 'right').drop('month_year',axis = 1)

    # df_econ.to_csv('../data/monthly_churn_with_econV2.csv')
    return df_econ


def final_processing(df):
    df['max_account'] = df.groupby('account_id')['end_of_month_balance'].transform(max)
    df['net_diff'] = df['monthly_deposit'] + df['monthly_withdrawal'] 

    df['start_month_balance'] = df['end_of_month_balance'] - df['net_diff']

    df['3month_rolling_sum'] = df.groupby('account_id')['net_diff'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['6month_rolling_sum'] = df.groupby('account_id')['net_diff'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
    df['12month_rolling_sum'] = df.groupby('account_id')['net_diff'].rolling(window=12, min_periods=1).sum().reset_index(level=0, drop=True)

    df['3month_rolling_max'] = df.groupby('account_id')['start_month_balance'].rolling(window=3, min_periods=1).max().reset_index(level=0, drop=True)
    df['6month_rolling_max'] = df.groupby('account_id')['start_month_balance'].rolling(window=6, min_periods=1).max().reset_index(level=0, drop=True)
    df['12month_rolling_max'] = df.groupby('account_id')['start_month_balance'].rolling(window=12, min_periods=1).max().reset_index(level=0, drop=True)

    df['three_monthly_decay'] = df.apply(lambda x: x['3month_rolling_sum']/x['3month_rolling_max'] if (x['3month_rolling_max'] >= 10) & (x['3month_rolling_sum'] != 0) else 0, axis=1)
    df['6_monthly_decay'] = df.apply(lambda x: x['6month_rolling_sum']/x['6month_rolling_max'] if (x['6month_rolling_max'] >= 10) & (x['6month_rolling_sum'] != 0) else 0, axis=1)
    df['12_monthly_decay'] = df.apply(lambda x: x['12month_rolling_sum']/x['12month_rolling_max'] if (x['12month_rolling_max'] >= 10) & (x['12month_rolling_sum'] != 0) else 0, axis=1)

    df['three_monthly_decay'] = df.apply(lambda x : x['three_monthly_decay'] if x['three_monthly_decay'] < 3 else 3,axis=1)
    df['6_monthly_decay'] = df.apply(lambda x : x['6_monthly_decay'] if x['6_monthly_decay'] < 3 else 3,axis=1)
    df['12_monthly_decay'] = df.apply(lambda x : x['12_monthly_decay'] if x['12_monthly_decay'] < 3 else 3,axis=1)

    df['UNEMP_rolling'] = df.groupby('account_id')['UNEMP'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # df['churn0.5'] = df.apply(lambda x: 1 if (x.three_monthly_decay < -0.5) | (x.churn == 1) else 0, axis=1)
    # # df['CHURN'] = df['churn0.5']

    # grouped = df.groupby('account_id')
    # df['CHURN'] = grouped['churn0.5'].shift(periods=-1,axis=0)
    # df['CHURN'] = df['CHURN'].fillna(0)

    df['age'] = (df['DATE'].dt.to_timestamp() - pd.to_datetime(df['dob'])).astype('timedelta64[Y]')

    cols_to_drop = ['DATE','creation_date','dob','total_cum_count_interactions'
                ,'no_of_monthly_deposits','no_of_monthly_withdrawals','monthly_interactions'
                ,'monthly_transactions','SENT','PSR','GDP/C','EXP/C','state','6month_rolling_sum',
                'monthly_deposit','6month_rolling_max','churn']

    df.drop(columns = cols_to_drop,axis=1,inplace=True)
    df_ = df.copy()
    del df
    # df_.to_csv('../../data/Modelling_DATA.csv')
    return df_

if __name__ == '__main__':

    cust_file_name = '../../raw_data/customers_tm1_e.csv'
    trans_file_name = '../../raw_data/transactions_tm1_e.csv'
    econ_data_file_name = '../../data/macroecon_features.csv'

    a = initial_processing(cust_file_name,trans_file_name)
    df_month = monthly_data(a,econ_data_file_name)
    df = final_processing(df_month)

#%%

 