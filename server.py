from flask import Flask
from flask import request
import json
import model

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    input = request.get_json(force=True)

    text = input['text']

    doc = {}
    doc['lang'] = input['lang']
    doc['tokenize'] = text
    doc['normalized'] = text

    result = model.predict(doc)

    if result == 1:
        return 'SPAM'
    else:
        return 'NON-SPAM'
