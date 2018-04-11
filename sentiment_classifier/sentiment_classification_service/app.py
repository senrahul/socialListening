#!venv/bin/python
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import logging
import operator 
import configuration as config
import classifier
logger = logging.getLogger("sentiment_classification_service")
hdlr = logging.FileHandler(config.LOG_FILE_NAME)
formatter = logging.Formatter(config.LOG_FORMAT)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

app = Flask(__name__, static_url_path='')

CORS(app)

model = classifier.Classifier()

@app.route('/query/<string:input_query>', methods=['GET', 'OPTIONS'])
def get_sentiment(input_query):
	categories = model.predict(input_query)
	return jsonify(sorted(dict(categories).items(), key = operator.itemgetter(1), reverse = True))


if __name__ == '__main__':
    app.run(host = config.host, port = config.port, debug=True)