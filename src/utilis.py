import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import pandas as pd

def save_object(file_path, object):
    with open(file_path, 'wb') as file:
        pickle.dump(object, file)

def model_performance(y_true, pred, model_name,model,features, irrelavent : int = 0):
    metrics_dict = {}
    mae = mean_absolute_error(y_true=y_true,y_pred=pred)
    mse = mean_squared_error(y_true=y_true,y_pred=pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true=y_true,y_pred=pred)
    adjusted_r2 = (1 - ((1-r2)*(features - 1))/(features-irrelavent-1))
    metrics_dict['mae'] = mae
    metrics_dict['mse'] = mse
    metrics_dict['rmse'] = rmse
    metrics_dict['r2'] = r2
    metrics_dict['adjusted_r2'] = adjusted_r2
    metrics_dict['model_name'] = model_name
    metrics_dict['model'] = model
    return metrics_dict

def best_model(performance_dict):
    performance_df = pd.DataFrame(performance_dict)
    performance_df.sort_values(by=['r2'],inplace=True,ascending=False)
    performance_df.to_csv('artifacts/performance_df.csv',header=True,index=False)
    return (performance_df['model_name'].values[0], performance_df['r2'].values[0],performance_df['model'].values[0])


    