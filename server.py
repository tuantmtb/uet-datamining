from flask import Flask
from flask import request
import model as model

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    input = request.get_json()

    text = input['text']

    doc = {}
    doc['lang'] = input['lang']
    doc['tokenize'] = text
    doc['normalized'] = text
    doc['label'] = 'SPAM'

    result = model.predict(doc)

    if result == 1:
        return 'SPAM'
    else:
        return 'NON-SPAM'
