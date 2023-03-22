import waitress
from flask import Flask,request
import logging
import classifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(module)s] %(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def hello():
    logger.info("Access to '/' endpoint")
    return f'This is the functionless home page\nSend a post request to /classify to get churn predictions'

@app.route('/classify',methods = ['POST'])
def classify():
    logger.info("Access to '/classify_one' endpoint")
    json_data = request.get_json()
    # logger.info(f'{json_data}')
    response = classifier.classify_document(json_data)
    return response

waitress.serve(app, port=5000, host='0.0.0.0')

