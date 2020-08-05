from pymongo import MongoClient
from keras.models import load_model
import tempfile
c = MongoClient('mongodb://exet4487:admin123@192.168.0.200:27017/jobs')
best_model = c['jobs']['jobs'].find_one({'exp_key': 'finaltestsubmit1'}, sort=[('result.loss', -1)])
temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
with open(temp_name, 'wb') as outfile:
    outfile.write(best_model['result']['model_serial'])
model = load_model(temp_name)
print(model)
print(model.summary())
print(c['jobs']['jobs'])
