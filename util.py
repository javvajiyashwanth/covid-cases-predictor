import json
import pickle
import numpy as np

__locations = None
__columns = None
__total_cases_model = None
__total_deaths_model = None

def predict(location, year, month, day):
    loc_index = __columns.index(location)
    
    x = np.zeros(len(__columns))
    x[0] = year
    x[1] = month
    x[2] = day

    if loc_index >= 0:
        x[loc_index] = 1

    return {
        'total_cases': int(__total_cases_model.predict([x])[0]*1000),
        'total_deaths': int(__total_deaths_model.predict([x])[0]*1000),
    }

def get_locations():
    return __locations

def load_data_and_model():
    global __locations
    global __columns
    global __total_cases_model
    global __total_deaths_model
    
    with open('C:/Users/yash/Desktop/Mini Project/data.json',) as f:
        __columns = json.load(f)['columns']
        __locations = __columns[3:]
    
    with open('C:/Users/yash/Desktop/Mini Project/total_cases_model.pickle', 'rb') as f:
        __total_cases_model = pickle.load(f)

    with open('C:/Users/yash/Desktop/Mini Project/total_deaths_model.pickle', 'rb') as f:
        __total_deaths_model = pickle.load(f)

if __name__ == "__main__":
    load_data_and_model()