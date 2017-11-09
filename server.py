from flask import Flask
from flask import request
import deep_model as model

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    text = request.form['text']

    doc = {}
    doc['lang'] = request.form['lang']
    doc['tokenize'] = text
    doc['normalized'] = text

    result = model.predict(doc)

    if result == 1:
        return 'SPAM'
    else:
        return 'NON-SPAM'
