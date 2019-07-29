# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://172.18.103.43:5000/predict"
JSON_PATH = "data/test_01.json"


payload = {"origin_json": open(JSON_PATH, 'r')}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    print(r)
    print("Request succeed")

# otherwise, the request failed
else:
    print("Request failed")
