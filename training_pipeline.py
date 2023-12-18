import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from src.utilis import model_performance
from sklearn.pipeline import Pipeline


@dataclass
class TrainingPipelineConfig:
    config_instance = os.path.join('artifacts','model.pkl')

class trainingpipeline:
    def __init__(self) -> None:
        self.training_pipeline_instance = TrainingPipelineConfig()

    def training(self,model_name,model,param_grid,X_train,y_train):
        pipeline = Pipeline([(model_name,model)])
        cv = GridSearchCV(estimator=pipeline,
                              param_grid = param_grid,
                              scoring = ['neg_mean_squared_error','r2'],
                              cv = 10,
                              refit = 'r2')
        cv.fit(X_train,y_train)
        return cv

    def grid_search(self, models : dict, param_grid : dict, data_dict : dict):

        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        model_performance_dict = {'model_name' : [],
                     'model' : [],
                     'mae' :[],
                     'mse' : [],
                     'rmse' : [],
                     'r2' : [],
                     'adjusted_r2' : []}
        for model in models.keys():
            print('Training: ' , model)
            cv = self.training(model_name=model,model=models[model], param_grid = param_grid[model], 
                               X_train = X_train, y_train = y_train)
            
            pred = cv.predict(X_test)

            metrics_dict = model_performance(y_true=y_test,pred=pred,model_name=model,model = cv,
                                             features=8,irrelavent=0)
            model_performance_dict['model_name'].append(model)
            model_performance_dict['mae'].append(metrics_dict['mae'])
            model_performance_dict['mse'].append(metrics_dict['mse'])
            model_performance_dict['rmse'].append(metrics_dict['rmse'])
            model_performance_dict['r2'].append(metrics_dict['r2'])
            model_performance_dict['adjusted_r2'].append(metrics_dict['adjusted_r2'])
            model_performance_dict['model'].append(cv)
        path = self.training_pipeline_instance.config_instance
        return model_performance_dict, path

