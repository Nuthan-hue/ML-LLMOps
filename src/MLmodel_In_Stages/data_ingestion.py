import os

import pandas as pd
dataset=pd.read_csv('https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv',sep=';')

def save_dataset(df, data_path='data'):
    data_path=os.path.join(data_path, 'raw')
    df.to_csv(data_path,'winequality-red.csv',index=False)


save_dataset(dataset,data_path='data')