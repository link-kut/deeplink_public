from flask import Flask, Response
from flask_restful import Resource, Api
from flask import request
import json

app = Flask(__name__)
api = Api(app)

class Ping(Resource):
    def get(self):
        return {'response': 'pong'}


class User(Resource):
    def post(self):
        # database insert
        return {'status': 'success'}

api.add_resource(Ping, '/ping')
api.add_resource(User, '/user')


data = {
    'hello': 'world',
    'number': 3
}

@app.route('/echo', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_echo():
    if request.method == 'GET':
        data_json = json.dumps(data)
        resp = Response(data_json, status=200, mimetype='application/json')
        resp.headers['Link'] = 'http://luisrei.com'
        return "ECHO: GET"
    elif request.method == 'POST':
        return "ECHO: POST"
    elif request.method == 'PUT':
        return "ECHO: PUT"
    elif request.method == 'DELETE':
        return "ECHO: DELETE"


if __name__ == '__main__':
    app.run(debug=True)