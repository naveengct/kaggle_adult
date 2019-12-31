from . import views
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def predict(X,temp,model):
    labelencoder = LabelEncoder()
    l=[1,2,4,5,6,7,11]
    for i in l:
        X[:,i]= labelencoder.fit_transform(X[:,i])
        temp[:,i]= labelencoder.transform((temp[:,i]))
    temp = np.asarray(temp, dtype=np.int32, order='C')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    temp=sc.transform(temp)
    
    result=model.predict(temp)
    return result