import pickle
import json
from keras.models import load_model

model = load_model('model/model_arm_20231119')

history = pickle.load(open('history/history_arm_20231119', "rb"))

    
hist_json_file='history/history_arm_json_20231119.json'
with open(hist_json_file,mode='w') as f:
    json.dump(history,f)
    