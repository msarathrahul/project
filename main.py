from src.logger import logging
from src.exception import CustomException
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
from training_pipeline import trainingpipeline
import pandas as pd
from src.utilis import best_model, save_object

data = DataIngestion()
train_path, test_path = data.dataingestion_instance(file_path = 'notebooks/data/concrete_data.csv')

transformation = DataTransformation()
data_dict = transformation.transform_data(train_path=train_path,test_path=test_path)

algorithms = ModelTrainer()
models = algorithms.models()
param_grid = algorithms.param_grid()

training = trainingpipeline()
performance_dict = training.grid_search(models=models,param_grid=param_grid,data_dict=data_dict)
path = performance_dict[1]
performance_dict = performance_dict[0]

model_name = best_model(performance_dict=performance_dict)
print("Best performing model :" , model_name[0])
print("R2 score :" , model_name[1])

save_object(file_path=path, object=model_name[2])