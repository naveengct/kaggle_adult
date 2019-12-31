from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from . import train1
from . import predict
# Create your views here.

def result(request):
    l3=list(request.GET['l3'].split(','))
    l3 = [' {0}'.format(elem) for elem in l3]
    l1=(request.GET['l1'])
    temp=np.array([l3])
    modulePath = os.path.dirname(__file__) 
    filePath = os.path.join(modulePath,'{0}/list.txt'.format(l1))
    f=open(filePath, "r")
    lines = f.read().splitlines()
    last_line = lines[-1]
    f.close()

    modulePath = os.path.dirname(__file__)  
    filePath = os.path.join(modulePath, 'data.pickle')
    data = pickle.load(open(filePath, 'rb'))

    filePath = os.path.join(modulePath, '{1}/{0}'.format(last_line,l1))
    model = pickle.load(open(filePath, 'rb'))
    
    modulePath = os.path.dirname(__file__) 
    dataset = pd.read_csv(os.path.join(modulePath, 'adult.csv'))

    X=data
    
    result=predict.predict(X,temp,model) 
    if result==[0]:
        result="<=50k"
    else:
        result=">50k"
    responseData = {
        "Result":result
    }
    return  JsonResponse(responseData,safe=False)

def train(request):
    l1=(request.GET['l1'])
    _datetime = datetime.now()
    datetime_str = _datetime.strftime("%Y-%m-%d-%H-%M-%S")
    
    classifier,responseData=train1.training(l1)

    modulePath = os.path.dirname(__file__) 
    
    filePath = os.path.join(modulePath,'{1}/model_{1}_{0}.pickle'.format(datetime_str,l1))
    pickle.dump(classifier, open(filePath, 'wb'))
    

    modulePath = os.path.dirname(__file__) 
    filePath = os.path.join(modulePath,'{0}/list.txt'.format(l1))
    f=open(filePath, "a+")
    f.write('model_{1}_{0}.pickle\r\n'.format(datetime_str,l1))
    f.close()
    return JsonResponse(responseData,safe=False)



