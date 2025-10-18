# Using mlflow through dagshub
# python ElasticNetWineModel_Dagshub.py,
# python ElasticNetWineModel_Dagshub.py 0.5.0.6

import sys
from urllib.parse import urlparse

import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import dagshub
dagshub.init(repo_owner='nuthan.maddineni23', repo_name='ML-LLMOps', mlflow=True)

dataset=pd.read_csv('https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv',sep=';')

train, test = train_test_split(dataset,test_size=0.2)
train_X = train.drop('quality',axis=1)
train_y = train[['quality']]
test_X = test.drop('quality',axis=1)
test_y = test[['quality']]

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01



with mlflow.start_run():

    Model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    Model.fit(train_X, train_y)

    predictions = Model.predict(test_X)

    ruse = root_mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    mae = mean_absolute_error(test_y, predictions)

    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("root_mean_squared_error", ruse)

    remote_server_url= 'https://dagshub.com/nuthan.maddineni23/ML-LLMOps.mlflow'

    mlflow.set_tracking_uri(remote_server_url)

    tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_uri_type_store != 'file':
        mlflow.sklearn.log_model(Model, 'model', registered_model_name='ElasticNetWineModel')
    else:
        mlflow.sklearn.log_model(Model, 'model')


