import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation  import cross_val_score
from sklearn.model_selection import train_test_split
import json
import os
    

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def updateFile(filename,key,cm):
    if not os.path.exists(filename):
        with open(filename,"w") as fp:
            tmp = {}
            json.dump(tmp,fp)
    with open(filename,'r') as fp:
        old = json.load(fp)
    if key not in old:
        old[key] = list()
    old[key].append(list(cm))
    with open(filename,'w') as fp:
        json.dump(old,fp,cls=MyEncoder)

def Bay(filename,X_train,X_test,y_train,y_test):
    print ("Bayes:")
    from sklearn.naive_bayes import MultinomialNB
    bay = MultinomialNB()
    bay_model = bay.fit(X_train,y_train)
    y_true, y_pred = y_test, bay_model.predict(X_test)
    print('Accuracy for test set: %f' % bay_model.score(X_test, y_test))
    cm = metrics.confusion_matrix(y_true, y_pred,labels=[0,1])
    print (confusion_matrix(y_true, y_pred,labels=[0,1]))
    print ('predict >1000, and true: %f' % (float(cm[1,1])/(cm[0,1]+cm[1,1]))) 
    print ('actual >1000, and predict true :%f'% (float(cm[1,1])/(cm[1,0]+cm[1,1]) ))
    updateFile(filename,"Bayes",cm)


def RandomF(filename,X_train,X_test,y_train,y_test):
    print ("RandomForest:")
    from sklearn.ensemble import RandomForestClassifier
    hotcounts = 0
    for y in y_train:
        if y == 1:
            hotcounts += 1
    ran = RandomForestClassifier(n_estimators=1000,max_features='auto', class_weight={1:len(y_train)/hotcounts, 0:len(y_train)/(len(y_train)-hotcounts)},n_jobs=3)
    ran_model = ran.fit(X_train,y_train)
    y_true, y_pred = y_test, ran_model.predict(X_test)
    print('Accuracy for test set: %f' % ran_model.score(X_test, y_test))
    cm = metrics.confusion_matrix(y_true, y_pred,labels=[0,1])
    print (confusion_matrix(y_true, y_pred,labels=[0,1]))
    print ('predict >1000, and true: %f' % (float(cm[1,1])/(cm[0,1]+cm[1,1]))) 
    print ('actual >1000, and predict true :%f'% (float(cm[1,1])/(cm[1,0]+cm[1,1]) ))
    updateFile(filename,"RandomForeast",cm)

def KNN(filename,X_train,X_test,y_train,y_test):
    print ("KNN:")
    knn = KNeighborsClassifier(n_neighbors = 3, algorithm='kd_tree',metric='euclidean')
    knn_model = knn.fit(X_train, y_train)
    y_true, y_pred = y_test, knn_model.predict(X_test)
    print('kdtree distance:manhattan accuracy for test set: %f' % knn_model.score(X_test, y_test))
    cm = metrics.confusion_matrix(y_true, y_pred,labels=[0,1])
    print (confusion_matrix(y_true, y_pred,labels=[0,1]))
    print ('predict >1000, and true: %f' % (float(cm[1,1])/(cm[0,1]+cm[1,1]))) 
    print ('actual >1000, and predict true :%f'% (float(cm[1,1])/(cm[1,0]+cm[1,1]) ))
    updateFile(filename,"KNN",cm)

def DT(filename,X_train,X_test,y_train,y_test):
    print ("DecisionTree:")
    from sklearn import tree
    Deci_Tree = tree.DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=15)
    Deci_model = Deci_Tree.fit(X_train,y_train)
    y_true, y_pred = y_test, Deci_model.predict(X_test)
    print('Accuracy for test set: %f' % Deci_model.score(X_test, y_test))
    cm = metrics.confusion_matrix(y_true, y_pred,labels=[0,1])
    print (confusion_matrix(y_true, y_pred,labels=[0,1]))
    print ('predict >1000, and true: %f' % (float(cm[1,1])/(cm[0,1]+cm[1,1]))) 
    print ('actual >1000, and predict true :%f'% (float(cm[1,1])/(cm[1,0]+cm[1,1]) ))
    updateFile(filename,"DT",cm)

def training(filename,X_train,X_test,y_train,y_test):
    Bay(filename,X_train,X_test,y_train,y_test)
    RandomF(filename,X_train,X_test,y_train,y_test)
    KNN(filename,X_train,X_test,y_train,y_test)
    DT(filename,X_train,X_test,y_train,y_test)

def round_training(filename):
    data = pd.read_csv(filename , sep = ',')
    bins = [-1,999,99999]
    name = [0,1]
    X = data.drop('likeCount' , 1).values 
    popular = pd.cut(data['likeCount'],bins,labels=name)
    data['popular'] = pd.cut(data['likeCount'],bins,labels=name)

    y = data.popular #data.likeCount
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    useful_feature = ExtraTreesClassifier()
    useful_feature = useful_feature.fit(X, y)
    model = SelectFromModel(useful_feature, prefit=True)
    X_new = model.transform(X)


    useful_feature = useful_feature.fit(X_new, y)
    # print (useful_feature.feature_importances_)
# 
    # for i in model.get_support([useful_feature]):
        # print (data.dtypes.index[i])
        
    X_newest = model.transform(X_new)
    useful_feature = useful_feature.fit(X_newest,y)
    print (useful_feature.feature_importances_)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    print ("Before select feature")
    training("nobefore.json",X_train,X_test,y_train,y_test)

    print ("")
    print ("after select feature")
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.5)
    training("noonce.json",X_train,X_test,y_train,y_test)

    print ("")
    print ("after select feature twice")
    X_train, X_test, y_train, y_test = train_test_split(X_newest, y, test_size=0.5)
    training("notwice.json",X_train,X_test,y_train,y_test)

for i in range(10):
    round_training('Data.csv')