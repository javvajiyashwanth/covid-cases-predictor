# Covid Cases Predictor
This is a web application that predicts the number of Covid-19 cases and deaths based on location (country) and date. The prediction is made using a linear regression machine learning model.

## Requirements
- Python 3
- Flask
- Pandas
- Numpy
- Scikit-learn

## Installation
- Clone the repository
<br />```git clone https://github.com/javvajiyashwanth/covid-cases-predictor.git```
- Install the requirements
<br />```pip install -r requirements.txt```
- Run the app
<br /> ```python app.py```
- Open the app on your browser by visiting http://localhost:5000/

## Data
The data used for building the model is taken from [Our World in Data](https://ourworldindata.org/coronavirus-data).

## Model
The model is built using linear regression. Training and testing data is split and the model is trained on the training data. The model is then used to make predictions on the testing data.

## Web App
A Flask web app is developed to make the request and get the results from the model and display them on the website. The user can select the location (country) and date and get the predicted number of cases and deaths.

## Note
This is a simple model built with available data. The predictions made by the model should be taken with a grain of salt as the actual numbers can vary greatly depending on various factors.
