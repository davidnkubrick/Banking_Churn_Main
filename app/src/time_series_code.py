"""
Time series module:
A time series regression model that predicts the what the balance for the customer is likely to be at the
end of next month. This value is then compared to our churn definition to classify them as churn or not churn.

time_series:

in - prediciton dataframe as df
out - Churn prediciton(classification), Balance prediction (Regression)
"""

import pandas as pd
from tqdm import tqdm 
import statsmodels.api as sm
import warnings

def time_series(df):

    warnings.filterwarnings('ignore')   #removing all warnings from the output

    df2 = df[['account_id','DATE','end_of_month_balance']]  #Stipping the large geature dataframe down into a dataframe with only time-series relevent features


    #Converting the multi-account dataset into a dictionary containing 1 dataframe containing transation data per customer ID (KEY)
    time_series = {}
    for idx,d in tqdm(df2.groupby('account_id'), disable=True) :

        idxs = d['DATE']
        time_series[idx] = d[['DATE', 'end_of_month_balance']] #d is the customer grouped dataframe and time_series is the customer keyed dictionary
        time_series[idx].index = pd.Index(idxs)
        time_series[idx] = time_series[idx].drop(['DATE'],axis =1)

    pred = []
    true = []
    cust = []
    for i in tqdm(time_series.keys(), disable=True) :
        #Try and except to attempt to make predictions on each customer providing they have enough data to predict on
        try :
            train_data = time_series[i][:str(time_series[i].index[-2])]   #Train set defined as all the data up to the final month     
            test_data = time_series[i][str(time_series[i].index[-1]):]    #Test set taken as the final month of transactional data

            model = sm.tsa.statespace.SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 2)) # SARIMA Model Definition - seasonal Autoregressive Integrated Moving Average
            results = model.fit()

            predictions = results.predict(str(time_series[i].index[-1]))       #predicting final month data for each customer

            pred.append(predictions[0])                                        #Predicted balance 
            true.append(test_data['end_of_month_balance'][0])                  #Actual Balance
            cust.append(i)                                                     #Customer ID
        except :
            pass

    pred2 = [i if i > 0 else 0 for i in pred]

    '''
    Bellow is a lot of general pandas cleaning and work to turn our balance prediction into a boolean classification of churn using our churn definition.

        -We defined churn as a balance decay of 50% occuring within a 3-month time period compared to the 3-month rolling maximum balance of the account.
        -By using our balance prediciton this value is calculated and stored as 'CURN_PRED' in the dataframe
        -Essentially we are looking at the final 3 values for each customer and calculating our 'three_mothly_decay' from them to classify churn
    '''

    df_no_last = df.drop(df.sort_values('DATE').groupby('account_id').tail(1).index, axis=0)
    df_right_cust = df_no_last[df_no_last['account_id'].isin(cust)][['DATE','account_id','end_of_month_balance']]
    df_right_cust = df_right_cust.sort_values('DATE').groupby('account_id').tail(2)

    true_churn = df.sort_values('DATE').groupby('account_id').tail(2)[['account_id','DATE']]
    true_churn_no_last = true_churn.drop(true_churn.sort_values('DATE').groupby('account_id').tail(1).index, axis=0)
    true_churn_no_last = true_churn_no_last[true_churn_no_last['account_id'].isin(cust)]

    pred_balance_df = pd.DataFrame({'account_id' : cust,'end_of_month_balance': pred2 } )
    for_dates = df[['DATE','account_id']].sort_values('DATE').groupby('account_id').tail(1)

    merge_date_pred = pd.merge(pred_balance_df,for_dates,on = 'account_id')

    concat_df = pd.concat([merge_date_pred[['account_id','end_of_month_balance','DATE']],df_right_cust]).sort_values(['account_id','DATE'])
    concat_df = pd.merge(concat_df,concat_df[['end_of_month_balance','account_id']].groupby('account_id').max()['end_of_month_balance'],on='account_id')
    concat_df.rename(columns = {'end_of_month_balance_x':'end_of_month_balance','end_of_month_balance_y':'3month_max'},inplace = True)

    diff = concat_df.groupby('account_id').tail(1)[['end_of_month_balance']].reset_index() - concat_df.groupby('account_id').head(1)[['end_of_month_balance']].reset_index()
    diff.drop('index',axis =1,inplace = True)
    diff.rename(columns = {'end_of_month_balance': 'net_diff'},inplace = True )
    diff['account_id'] = cust
    concat_df = pd.merge(concat_df,diff,on='account_id')

    concat_df['three_monthly_decay'] = concat_df.apply(lambda x: x['net_diff']/x['3month_max'] if (x['3month_max'] >= 10) & (x['net_diff'] != 0) else 0, axis=1)
    concat_df['three_monthly_decay'] = concat_df.apply(lambda x : x['three_monthly_decay'] if x['three_monthly_decay'] < 3 else 3,axis=1)

    concat_df[['DATE','three_monthly_decay','account_id']]
    df_no_last2 = concat_df[['DATE','three_monthly_decay','account_id']].sort_values('DATE').groupby('account_id').tail(1)
    df_no_last2['CHURN_PRED'] = df_no_last2.apply(lambda x : 1 if x['three_monthly_decay'] < -0.5 else 0,axis=1)

    return df_no_last2['CHURN_PRED'], pred    #Returning classifiction predicitons and regression predictions
