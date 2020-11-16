import json
import pickle
import numpy as np

__data = None
__columns = None
__new_cases_model = None
__total_cases_model = None
__new_deaths_model = None
__total_deaths_model = None

def predict(continent, location, date):
    con_index = __columns.index(continent)
    loc_index = __columns.index(location)
    
    x = np.zeros(len(__columns))
    x[0] = __data[continent][location]
    x[1] = date

    if loc_index >= 0:
        x[loc_index] = 1

    if con_index >= 0:
        x[con_index] = 1

    return {
        'new_cases': int(__new_cases_model.predict([x])[0]),
        'total_cases': int(__total_cases_model.predict([x])[0]),
        'new_deaths': int(__new_deaths_model.predict([x])[0]),
        'total_deaths': int(__total_deaths_model.predict([x])[0]),
    }

def get_data():
    return __data

def load_data_and_model():
    global __data
    global __columns
    global __new_cases_model
    global __total_cases_model
    global __new_deaths_model
    global __total_deaths_model
    
    with open('C:/Users/yash/Desktop/Mini Project/data.json',) as f:
        temp = json.load(f)
        __data = temp['data']
        __columns = temp['columns']

    with open('C:/Users/yash/Desktop/Mini Project/new_cases_model.pickle', 'rb') as f:
        __new_cases_model = pickle.load(f)
    
    with open('C:/Users/yash/Desktop/Mini Project/total_cases_model.pickle', 'rb') as f:
        __total_cases_model = pickle.load(f)
    
    with open('C:/Users/yash/Desktop/Mini Project/new_deaths_model.pickle', 'rb') as f:
        __new_deaths_model = pickle.load(f)

    with open('C:/Users/yash/Desktop/Mini Project/total_deaths_model.pickle', 'rb') as f:
        __total_deaths_model = pickle.load(f)

if __name__ == "__main__":
    load_data_and_model()
    print(predict('Asia', 'Afghanistan', 1586131200))