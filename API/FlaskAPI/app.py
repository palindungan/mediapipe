from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api

# create an instance of flask
app = Flask(__name__)
# creating an API object
api = Api(app)


# For GET request to http://localhost:5000/
class GetData(Resource):
    def get(self):
        return {"data": 'success'}, 200


# For Post request to http://localhost:5000/employee
class AddData(Resource):
    def post(self):
        if request.is_json:
            # return a json response
            return make_response(jsonify({'Id': 1, }), 201)
        else:
            return {'error': 'Request must be JSON'}, 400


# For put request to http://localhost:5000/update/?
class UpdateData(Resource):
    def put(self, idx):
        if request.is_json:
            data = 1
            if data is None:
                return {'error': 'not found'}, 404
            else:
                return f'{idx} is updated', 200
        else:
            return {'error': 'Request must be JSON'}, 400


# For delete request to http://localhost:5000/delete/?
class DeleteData(Resource):
    def delete(self, idx):
        data = 1
        if data is None:
            return {'error': 'not found'}, 404
        return f'{idx} is deleted', 200


api.add_resource(GetData, '/')
api.add_resource(AddData, '/add')
api.add_resource(UpdateData, '/update/<int:id>')
api.add_resource(DeleteData, '/delete/<int:id>')

#
if __name__ == '__main__':
    app.run(debug=True)
