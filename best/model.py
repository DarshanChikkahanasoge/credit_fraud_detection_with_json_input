import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

data=pd.read_csv('best/creditcard.csv')
num_pipeline=Pipeline([
    ('std_scaler',StandardScaler())
])

data['Time']=num_pipeline.fit_transform(data['Time'].values.reshape(-1,1))
data['Amount']=num_pipeline.fit_transform(data['Amount'].values.reshape(-1,1))

x=data.iloc[:,data.columns!='Class']
y=data.iloc[:,data.columns=='Class']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)

ytrain=np.ravel(ytrain)
ytest=np.ravel(ytest)

clf=LogisticRegression()
clf.fit(xtrain,ytrain)

pickle.dump(clf,open('best/model.pkl','wb'))

with open("best/model.bin",'wb') as f_out:
    pickle.dump(clf,f_out)
    f_out.close()

