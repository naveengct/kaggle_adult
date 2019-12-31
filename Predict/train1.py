from . import views
from . import grid_select
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
def training(l1):
    modulePath = os.path.dirname(__file__) 
    dataset = pd.read_csv(os.path.join(modulePath, 'adult.csv'))
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 14].values

    X=X[:,[0,1,3,4,5,6,7,9,10,11,12,13]]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = ' ?', strategy = 'most_frequent')
    for i in range(11):
        temp=X[:,i].reshape(len(X[:,i]), 1)
        imputer = imputer.fit(temp)
        temp = imputer.transform(temp)
        temp=temp.reshape(len(temp),-1)
        X[:,i]=temp[:,0]

    filePath = os.path.join(modulePath,'data.pickle')
    pickle.dump(X, open(filePath, 'wb'))
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    y= labelencoder.fit_transform(y)
    l=[1,2,4,5,6,7,11]
    for i in l:
        X[:,i]= labelencoder.fit_transform(X[:,i])
    #import pdb;pdb.set_trace()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X = sc.transform(X)
    
    if l1=="Logistic_Regression":
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train,y_train)
    if l1=="SVM":
        from sklearn.svm import SVC
        classifier = SVC()
        grid_select.grid_SVM(classifier)
        classifier.fit(X_train,y_train)
    if l1=="KNN":
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
        grid_select.grid_KNN(classifier)
        classifier.fit(X_train,y_train)
    if l1=="Naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
    if l1=="Random_Forest":
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
        grid_select.grid_RFA(classifier)
        classifier.fit(X_train,y_train)
    if l1=="XG_Boost":
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        grid_select.grid_XG(classifier)
        classifier.fit(X_train, y_train)
    

    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acc_scr=accuracy_score(y_test,y_pred)
    recall=float(cm[0][0]/(cm[0][0]+cm[1][0]))
    precision=float(cm[0][0]/(cm[0][0]+cm[0][1]))

    responseData={
        "Status ":"Model created sucessfully",
        " Model Type":"{0}".format(l1),
        " Model Name":'model_{0}.pickle'.format(l1),
        " Accuracy":acc_scr,
        " Recall":recall,
        " Precision":precision
    }

    return classifier,responseData
