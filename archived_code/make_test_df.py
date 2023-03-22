#%%
import pandas as pd
import numpy as np

#%%

def make_test_df(cust,trans,samp):
    cust = pd.read_csv(cust)
    trans = pd.read_csv(trans)
    samp = pd.read_csv(samp)

    # customer_ids = samp.account_id

    new_trans = trans[trans.account_id.isin(samp.account_id)]

    customer_id = new_trans.customer_id

    new_cust = cust[cust.customer_id.isin(customer_id)]


    return new_cust,new_trans


if __name__ == '__main__':
    cust_file_name = 'raw_data/customers_tm1_e.csv'
    trans_file_name = 'raw_data/transactions_tm1_e.csv'
    samp = 'test_data/submission_sample.csv'
    new_cust,new_trans = make_test_df(cust_file_name,trans_file_name,samp)


    new_cust.to_csv('test_data/cust_test.csv')
    new_trans.to_csv('test_data/trans_test.csv')

#%%