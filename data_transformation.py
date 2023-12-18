from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
from src.utilis import save_object

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformationConfig:
    config_instance = os.path.join('artifacts','preprocessor_pipeline.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.datatransformation_instance = DataTransformationConfig()

    def transformers(self):
        column_transformer_1 = ColumnTransformer([('standardize',StandardScaler(),list(range(0,8)))]
                                                 ,remainder='passthrough')
        column_transformer_2 = ColumnTransformer([('transformation',PowerTransformer(method='yeo-johnson'),list(range(0,8)))],
                                                 remainder='passthrough')
        
        pipeline = Pipeline([('ct_1',column_transformer_1),
                  ('ct_2',column_transformer_2)])
        
        return pipeline
    
    def transform_data(self, train_path, test_path):

        features = ['cement','blast_furnace_slag','fly_ash','water',
                    'superplasticizer','coarse_aggregate','fine_aggregate ','age']
        target = ['concrete_compressive_strength']
        pipeline = self.transformers()
        save_object(self.datatransformation_instance.config_instance, pipeline)

        train_data  = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        X_train, y_train = train_data[features], train_data[target]
        X_test, y_test = test_data[features], test_data[target]

        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)

        return {'X_train' : X_train, 'y_train' : y_train,
                 'X_test' : X_test, 'y_test' : y_test}