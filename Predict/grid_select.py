from . import train1
from . import views
from sklearn.model_selection import GridSearchCV

def grid_SVM(classifier):
    kernel=['linear','rbf','poly','sigmoid']
    GridSearchCV(estimator=classifier,cv=3,param_grid=dict(kernel=kernel,random_state=[0]))

def grid_KNN(classifier):
    neighbor=[10]
    metric=['minkowski']
    p=[2]
    grid=GridSearchCV(estimator=classifier,cv=3,param_grid=dict(n_neighbors =neighbor, metric = metric, p = p))

def grid_RFA(classifier):
    est=[10,20,30]
    cri=['entropy','gini']
    grid=GridSearchCV(estimator=classifier,cv=3,param_grid=dict(n_estimators = est, criterion = cri, random_state = [0]))

def grid_XG(classifier):
    boost=['gbtree','gblinear','dart']
    grid=GridSearchCV(estimator=classifier,cv=3,param_grid=dict(booster=boost))
