import pickle
import json
import numpy
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import time

def init():
    global model
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('LOS_RF_model2.pkl')
    # deserialize the model file back into a sklearn model
    model = pickle.load(open(model_path, 'rb'))
    global inputs_dc, prediction_dc
    # this setup will help us save our inputs under the "inputs" path in our Azure Blob
    inputs_dc = ModelDataCollector(model_name="LOS_RF_model2.pkl", identifier="inputs", feature_names=['rcount', 'gender', 'dialysisrenalendstage', 'asthma',
       'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
      'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration',
       'secondarydiagnosisnonicd9',  'fid', 'Capacity', 'Name', 'daysofweek_admit'])
    # this setup will help us save our ipredictions under the "predictions" path in our Azure Blob
    prediction_dc = ModelDataCollector("LOS_RF_model2.pkl", identifier="predictions", feature_names=["prediction1"]) 


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
        inputs_dc.collect(data) #this call is saving our input data into our blob
        prediction_dc.collect(result)#this call is saving our prediction data into our blob
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
