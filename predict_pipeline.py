from src.logger import logging
from src.exception import CustomException
import sys
import pickle
import numpy as np
import pandas as pd

with open('artifacts/preprocessor_pipeline.pkl','rb') as obj:
    preprocessor_pipeline = pickle.load(obj)

with open('artifacts/model.pkl','rb') as obj:
    model = pickle.load(obj)

array = np.array([float(i) for i in sys.argv[1:]]).reshape(1,-1)
features = ['cement','blast_furnace_slag','fly_ash','water',
            'superplasticizer','coarse_aggregate','fine_aggregate ','age']

df = pd.DataFrame(array,columns = features)
array = preprocessor_pipeline.transform(df)
#df = pd.DataFrame(array,columns = features)
prediction = model.predict(array)
print('Concrete compressive strength : ',end='')
print(prediction[0])