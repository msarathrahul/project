import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utilis import save_object

import pandas as pandas
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    config_models = os.path.join('artifacts','models.pkl')
    config_param_grid = os.path.join('artifacts','param_grid.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.modeltrainier_instance = ModelTrainerConfig()

    def models(self):
        models = {'Linear Regression' : LinearRegression(),
                'Ridge Regression' : Ridge(),
                'Lasso Regression' : Lasso(),
                'Decision tree Regression' : DecisionTreeRegressor(),
                'SVM Regression' : SVR(),
                'KNN Regression' : KNeighborsRegressor(),
                'Random Forest Regression' : RandomForestRegressor(),
                'AdaBoost Regression' : AdaBoostRegressor(),
                'GradientBoosting Regression' : GradientBoostingRegressor(),
                'XGB Regression' : XGBRegressor()}
        save_object(file_path=self.modeltrainier_instance.config_models,object=models)
        return models
    
    def param_grid(self):
        param_grid = {
            'Linear Regression': {},
            
            'Ridge Regression': {
                'Ridge Regression__alpha': [0.1, 1, 10]
            },
            
            'Lasso Regression': {
                'Lasso Regression__alpha': [0.1, 1, 10]
            },
            
            'Decision tree Regression': {
                'Decision tree Regression__max_depth': [None, 10, 20],
                'Decision tree Regression__min_samples_split': [2, 5, 10]
            },
            
            'SVM Regression': {
                'SVM Regression__C': [0.1, 1, 10],
                'SVM Regression__kernel': ['linear', 'rbf']
            },
            
            'KNN Regression': {
                'KNN Regression__n_neighbors': [3, 5, 7],
                'KNN Regression__weights': ['uniform', 'distance']
            },
            
            'Random Forest Regression': {
                'Random Forest Regression__n_estimators': [50, 100, 200],
                'Random Forest Regression__max_depth': [None, 10, 20],
                'Random Forest Regression__min_samples_split': [2, 5, 10]
            },
            
            'AdaBoost Regression': {
                'AdaBoost Regression__n_estimators': [50, 100, 200],
                'AdaBoost Regression__learning_rate': [0.01, 0.1, 1]
            },
            
            'GradientBoosting Regression': {
                'GradientBoosting Regression__n_estimators': [50, 100, 200],
                'GradientBoosting Regression__learning_rate': [0.01, 0.1, 1],
                'GradientBoosting Regression__max_depth': [3, 5, 10]
            },
            
            'XGB Regression': {
                'XGB Regression__n_estimators': [50, 100, 200],
                'XGB Regression__learning_rate': [0.01, 0.1, 1],
                'XGB Regression__max_depth': [3, 5, 10]
            }
        }
        save_object(self.modeltrainier_instance.config_param_grid,object=param_grid)
        return param_grid