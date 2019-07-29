# coding=utf-8
import flask
import tensorflow as tf
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
from models.bert.bert_model import BertModel

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
graph = []
model = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global graph
    model = BertModel()
    graph = tf.get_default_graph()


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("origin_json"):
            origin_json = json.load(flask.request.files["origin_json"])
        else:
            origin_json = flask.request.get_json()

        global graph
        with graph.as_default():
            preds = model.predict([origin_json['sentence1']], [origin_json['sentence2']])
        data["predictions"] = float(preds[0][0])
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host="0.0.0.0", port=5000)
