# File Explanation

The main.py file runs the app on localhost:5000. To receive a prediction, a request must be sent.

The file, example_request.py, shows an example of the request that must be sent. The required input is a csv that contains a colulmn name 'account_id'. The file_path to this csv must be passed. This csv is then converted to json and sent in the post request.  A flag can be passed into this file stating whther time-series should be used or not.

An example post request using this file and with time series being used as the predictor looks like:

``` bash 
python example_request.py --time-series True
```

Main.py takes the json from the post request and sends it to the classifier.py file. 

The classifier.py file takes the json from main.py and converts it into a dataframe. It then calls functions from data_generation.py to create the appropriate features that are required for input into the model. The dataset is then predicted on, with a json being output. The json has information on the number of samples being predicted, the customer_id, date and final prediciton for each sample.

For classifier.py to work, it needs to load in a model and transformer. This have already been created from train.py and saved in the resources/model folder.

# Docker instructions:

To build the docker image, run the following command from the 'app' folder:
``` bash
docker build -f Dockerfile -t bank_churn_app:latest .
```

To run the the app on docker, run the line:
``` bash
docker run -it -p 5000:5000 bank_churn_app
```


If a daemon error occurs, it may be because docker desktop is not open so open it and try again.

The purpose of -p 5000:5000 is to sync local host port 5000 to the container port 5000, which the app is being served on, specified in main.py. I believe that we can't access the app locally without adding this.

To send the image to somebody, the image needs to be converted to .tar format and then compressed. This can be done by running the following commands:

```bash
docker save -o ~/Downloads/bank_churn_app.tar bank_churn_app

tar -czvf bank_churn_app.tar.gz bank_churn_app.tar
```

Note that that the second line should be run from within the location of the .tar file, which is the Downloads folder in this case.

The app can then be loaded using the command:

```bash
tar -xzf bank_churn_app.tar\ \(1\).gz
docker load -i bank_churn_app.tar
```