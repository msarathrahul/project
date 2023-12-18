from src.logger import logging
from src.exception import CustomException

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw_data.csv')
    train_data_path = os.path.join('artifacts','train_data.csv')
    test_data_path = os.path.join('artifacts','test_data.csv')

class DataIngestion:
    def __init__(self):
        self.config_instance = DataIngestionConfig()
    
    def dataingestion_instance(self, file_path):
        os.makedirs(os.path.dirname(self.config_instance.raw_data_path),exist_ok=True)

        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=0.3, random_state=42)

        df.to_csv(self.config_instance.raw_data_path,index = False,header=True)
        train.to_csv(self.config_instance.train_data_path,index = False,header = True)
        test.to_csv(self.config_instance.test_data_path,index = False,header = True)

        train_path = self.config_instance.train_data_path
        test_path = self.config_instance.test_data_path

        return (train_path, test_path)


    