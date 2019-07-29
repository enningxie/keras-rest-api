# coding=utf-8
from threading import Thread
import numpy as np
import flask
import redis
import uuid
import time
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
import tensorflow as tf
from models.bert.bert_model import BertModel

# initialize constants used for server queuing
JSON_QUEUE = "json_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None


def similar_process():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    print("* Loading model...")
    model = BertModel()
    graph = tf.get_default_graph()
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(JSON_QUEUE, 0, BATCH_SIZE - 1)
        jsonIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            tmp_json = [[q['sentence1']], [q['sentence2']]]

            # check to see if the batch list is None
            if batch is None:
                batch = tmp_json

            # otherwise, stack the data
            else:
                batch = np.vstack([batch, tmp_json])

            # update the list of image IDs
            jsonIDs.append(q["id"])

        # check to see if we need to process the batch
        if len(jsonIDs) > 0:
            # classify the batch
            with graph.as_default():
                preds = model.predict(batch[0], batch[1])

            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(jsonIDs, preds):
                # initialize the list of output predictions
                output = []

                # loop over the results and add them to the list of
                # output predictions
                r = {"probability": float(resultSet[0])}
                output.append(r)

                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

            # remove the set of images from our queue
            db.ltrim(JSON_QUEUE, len(jsonIDs), -1)

        # sleep for a small amount
        time.sleep(SERVER_SLEEP)


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

        # generate an ID for the classification then add the
        # classification ID + image to the queue
        k = str(uuid.uuid4())
        origin_json['id'] = k
        db.rpush(JSON_QUEUE, json.dumps(origin_json))

        # keep looping until our model server returns the output
        # predictions
        while True:
            # attempt to grab the output predictions
            output = db.get(k)

            # check to see if our model has classified the input
            # image
            if output is not None:
                # add the output predictions to our data
                # dictionary so we can return it to the client
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                # delete the result from the database and break
                # from the polling loop
                db.delete(k)
                break

            # sleep for a small amount to give the model a chance
            # to classify the input image
            time.sleep(CLIENT_SLEEP)

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print("* Starting model service...")
    t = Thread(target=similar_process, args=())
    t.daemon = True
    t.start()

    # start the web server
    print("* Starting web service...")
    app.run(host="0.0.0.0")
