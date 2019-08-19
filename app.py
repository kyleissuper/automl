from flask import Flask, request
from gevent.pywsgi import WSGIServer
from autokeras import TextClassifier
import settings

import json


app = Flask(__name__)
app.debug = False


@app.route("/predict", methods=["POST"])
def predict():
    input_string = request.form["message"]
    if input_string == "test":
        prediction_string = ""
    else:
        prediction_number = clf.predict([input_string])[0]
        prediction_string = prediction_map[prediction_number]
    return json.dumps({"Prediction": prediction_string})

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    clf = TextClassifier(verbose=False)
    clf.num_labels = 3
    clf.output_model_file = "data/v2.h5"
    
    prediction_map = {
            0: "do_not_reply",
            1: "workbot",
            2: "knowledge_base"
            }

    http_server = WSGIServer(("", settings.PORT), app)
    http_server.serve_forever()
