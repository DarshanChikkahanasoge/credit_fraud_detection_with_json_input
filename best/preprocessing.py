from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle


data=pd.read_csv('best/creditcard.csv')

num_pipeline=Pipeline([
    ('std_scaler',StandardScaler())
])

num_pipeline.fit(data['Amount'].values.reshape(-1,1))

def predict_out(config):
    if type(config)==dict:
        df=pd.DataFrame(config)
    else:
        df=config

    df['Time']=num_pipeline.transform(df['Time'].values.reshape(-1,1))
    df['Amount']=num_pipeline.transform(df['Amount'].values.reshape(-1,1))

    return df




