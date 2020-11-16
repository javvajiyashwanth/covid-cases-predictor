from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

@app.route('/get_data')
def get_data():
    response = jsonify({
        'data': util.get_data()
    })
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    continent = request.form['continent']
    location = request.form['location']
    date = int(request.form['date'])

    response = jsonify({
        'prediction': util.predict(continent, location, date)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print('server starting')
    util.load_data_and_model()
    app.run()