from src.logger import logging
from src.exception import CustomException
from data_ingestion import DataIngestion
from data_transformation import DataTransformation

data = DataIngestion()
train_path, test_path = data.dataingestion_instance(file_path = 'notebooks/data/concrete_data.csv')

transformation = DataTransformation()
data_dict = transformation.transform_data(train_path=train_path,test_path=test_path)