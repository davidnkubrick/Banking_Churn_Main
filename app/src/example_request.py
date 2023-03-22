'''
Request.py

First file in the pipeline that allows the user to recieve predictions for their local 'submission_sample.csv' via the stacked or time series model.
'''
# %% --------------------------------------------------------------------------
# 1: Imports
# -----------------------------------------------------------------------------

import requests
import json
import argparse
import pandas as pd

# %% --------------------------------------------------------------------------
# 2: Parsing time series flag
# -----------------------------------------------------------------------------

#Added a --time-series flag allowing the user to request a time series prediction when sending their request
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--time-series', type=str, default='False', metavar='T',
                        help='Choose whether to run a time_series model or ordinary ML model')
args = parser.parse_args()

# %% --------------------------------------------------------------------------
# 3: Importing prediciton files
# -----------------------------------------------------------------------------

# Read CSV data from files
test_id_csv = '../resources/data/submission_sample.csv'

with open(test_id_csv, "r") as f:
    samp_data = [line.strip().split(",") for line in f.readlines()]


# %% --------------------------------------------------------------------------
# 4: Generating JSON POST
# -----------------------------------------------------------------------------

# Create JSON object
json_data = {"data": samp_data}

new_json_data = {
    "data": {"samp": {"column_names": json_data["data"][0],
                      "data": json_data["data"][1:]
                      }
             }
    ,"flags": {'time_series': args.time_series}
}


# Serialize to JSON format
json_str = json.dumps(new_json_data)

# %% --------------------------------------------------------------------------
# 5: Sending POST to app
# -----------------------------------------------------------------------------

# The app is hosted locally and is run via main.py
if __name__ == '__main__':
    # Send POST request with JSON payload
    url = "http://localhost:5000/classify"
    headers = {"Content-Type": "application/json"}
    post_response = requests.post(url, data=json_str, headers=headers)
    if post_response.status_code == 200:            # Status code 200 indicating that the post request has succeeded
        json_response = post_response.json()
        # print(json_response)
        response_df = pd.DataFrame(json_response['results']['predictions'])
        print(response_df)
    else:
        print("Error: Status code", post_response.status_code)
