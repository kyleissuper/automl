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
    prediction_number = clf.predict([input_string])[0]
    prediction_string = prediction_map[prediction_number]
    return json.dumps({"Prediction": prediction_string})

if __name__ == "__main__":
    clf = TextClassifier(verbose=False)
    clf.num_labels = 5
    clf.output_model_file = "v2.h5"
    
    prediction_map = {
            0: "workbot",
            1: "crm",
            2: "do_not_reply",
            3: "knowledge_base",
            4: "troubleshoot"
            }

    http_server = WSGIServer(("", settings.PORT), app)
    http_server.serve_forever()
