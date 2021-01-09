from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

@app.route('/get_locations')
def get_locations():
    response = jsonify({
        'locations': util.get_locations()
    })
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']

    response = jsonify({
        'prediction': util.predict(location, year, month, day)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print('server starting')
    util.load_data_and_model()
    app.run()