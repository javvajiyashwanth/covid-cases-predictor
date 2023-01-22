import os
from flask import Blueprint, redirect, render_template, request, json, url_for
from flask_wtf import FlaskForm
from wtforms import (DateField, SelectField)
from wtforms.validators import InputRequired
import pickle
import numpy as np

models_path = os.path.join(os.path.dirname(__file__), 'ml_models')
totalCases = pickle.load(open(models_path + '/total_cases_model.pickle', 'rb'))
totalDeaths = pickle.load(
    open(models_path + '/total_deaths_model.pickle', 'rb'))


views = Blueprint('views', __name__, static_folder="static")

data = json.load(open(os.path.join(views.static_folder, 'data.json')))
columns = data['columns']
locations = columns[3:]


class PredictionForm(FlaskForm):
    location = SelectField('Location', choices=locations,
                           validators=[InputRequired()])
    date = DateField('Date', validators=[InputRequired()])


def getPrediction(location, year, month, day):
    loc_index = columns.index(location)

    x = np.zeros(len(columns))
    x[0] = year
    x[1] = month
    x[2] = day

    if loc_index >= 0:
        x[loc_index] = 1

    return {
        'total_cases': int(totalCases.predict([x])[0] * 1000),
        'total_deaths': int(totalDeaths.predict([x])[0] * 1000),
    }


@views.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    if form.validate_on_submit():
        location = form.location.data
        date = form.date.data
        response = getPrediction(location, date.year, date.month, date.day)
        response['location'] = location
        response['date'] = date
        return render_template('result.html', response = response)
    return render_template('predict.html', form = form)


@views.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html', response = None)