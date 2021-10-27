import json

from flask import Flask
from flask_cors import CORS
from flask import request

from blueprints.graph import graph_bp as graph
from blueprints.records import records

from models.submodels.AE_model import AE

app = Flask(__name__)
CORS(app)

app.register_blueprint(graph, url_prefix='/graph')
app.register_blueprint(records, url_prefix='/records')


@app.route('/', methods=['GET'])
def get_connection():
    return 'Connection'


if __name__ == '__main__':
    # app.run(host="0.0.0.0")
    app.run(host="127.0.0.1", port=5001)
