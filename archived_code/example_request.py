#%%
import requests
import json
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--time-series', type=str, default='False', metavar='T',
                            help='Choose whether to run a time_series model or ordinary ML model')
    args = parser.parse_args()

# Read CSV data from files
cust_filename = "../resources/data/samp_cust_test.csv"
trans_filename = "../resources/data/samp_trans_test.csv"

# cust_filename = '../../test_data/cust_test.csv'
# trans_filename = '../../test_data/trans_test.csv'
# sample_test = '../resources/data/submission_sample.csv'

with open(cust_filename, "r") as f:
    cust_data = [line.strip().split(",") for line in f.readlines()]

with open(trans_filename, "r") as f:
    trans_data = [line.strip().split(",") for line in f.readlines()]

# Create JSON object
data_dict = {"cust": cust_data, "trans": trans_data}
json_data = {"data": data_dict}

new_json_data = {
    "data": {"cust": {"column_names": json_data["data"]["cust"][0],
                      "data": json_data["data"]["cust"][1:]
                      },
             "trans":  {"column_names": json_data["data"]["trans"][0],
                        "data": json_data["data"]["trans"][1:]
                        }
             }
    # ,"flags": {'time_series': args.time_series}
}


# Serialize to JSON format
json_str = json.dumps(new_json_data)

if __name__ == '__main__':
    # Send POST request with JSON payload
    url = "http://localhost:5000/classify"
    headers = {"Content-Type": "application/json"}
    post_response = requests.post(url, data=json_str, headers=headers)
    if post_response.status_code == 200:
        json_response = post_response.json()
        # print(json_response)
        response_df = pd.DataFrame(json_response['results']['predictions'])
        # print(response_df)
    else:
        print("Error: Status code", post_response.status_code)