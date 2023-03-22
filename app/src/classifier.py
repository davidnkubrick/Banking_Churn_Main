#%%
import joblib
import logging
import json
import pandas as pd
import numpy as np
from data_generation import initial_processing, monthly_data, final_processing, make_test_df
import warnings
from time_series_code import time_series

#%%
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

model = joblib.load('../resources/model/modelV6.joblib')
transformer = joblib.load('../resources/model/transformerV6.joblib')

def build_response(n:int):
    predictions = []
    for _ in range(n):
        predictions.append({
        })
    my_json = {
        'results': {
            'count': n,
            'predictions': predictions
        }
    }
    return my_json

def classify_document(json_data):
    logger.info('Access to classify_document()')

    samp_df = pd.DataFrame(json_data['data']['samp']['data'],columns=json_data['data']['samp']['column_names'])
    cust_df,trans_df = make_test_df(samp_df)
    econ_data_file_name = '../resources/data/macroecon_features.csv'
    df_month = monthly_data(initial_processing(cust_df,trans_df),econ_data_file_name)
    account_id = df_month.account_id

    # is_time_series = 'True'
    is_time_series = json_data['flags']['time_series']

    if is_time_series == 'True':
        subset = df_month[:int(round(len(df_month)/10,0))]
        y_pred,balance_pred = time_series(subset)
        response = build_response(len(y_pred))

        for index,(y_value,balance_value) in enumerate(zip(y_pred,balance_pred)):
            logger.debug('Populating the response object')
            response["results"]["predictions"][index]["sample_num"] = str(index+1)+'/'+str(len(y_pred))
            response["results"]["predictions"][index]["account_id"] = str(np.unique(account_id)[index])
            response["results"]["predictions"][index]["prediction"] = round(float(y_value),3)
            response["results"]["predictions"][index]["predicted_balance"] = round(float(balance_value),2)
            
    else:
        df_final = final_processing(df_month)
        logger.info('Transforming the data')
        df_final['account_id'] = account_id

        last_samp = df_final.groupby('account_id').tail(1)

        last_samp.drop('account_id',axis=1,inplace=True)

        data_trans = transformer.transform(last_samp)

        logger.debug('Predicting the category for the document')
        
        y_pred = model.predict_proba(data_trans)[:,1]

        response = build_response(len(y_pred))
        for index,value in enumerate(y_pred):
            logger.debug('Populating the response object')
            response["results"]["predictions"][index]["sample_num"] = str(index+1)+'/'+str(len(y_pred))
            response["results"]["predictions"][index]["acount_id"] = int(account_id[index])
            response["results"]["predictions"][index]["predict_probability"] = float(value)
            response["results"]["predictions"][index]["prediction"] = round(float(value),0)

    return json.dumps(response)



if __name__ == '__main__':
    from example_request import *

    json_data = new_json_data
    response = classify_document(json_data)
    response_df = pd.DataFrame(json.loads(response)['results']['predictions'])
    submission = pd.read_csv('../resources/data/submission_sample.csv')
    submission['pred_churn'] = response_df.predict_probability
    # submission.to_csv('../../test_data/submission_V4_denmark.csv', index=False)

#%%
