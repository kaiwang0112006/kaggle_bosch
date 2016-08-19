########################################################
#
#  use numeric data only
#
########################################################

import pandas as pd
from sklearn import cross_validation
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from submit import *

train_numeric = pd.read_csv('train_numeric.csv')
train_date = pd.read_csv('train_date.csv')
data_merge = pd.merge(train_numeric, train_date, on='Id',suffixes=('trnum', 'trdate'))

dataclean = data_merge.dropna(axis=1, thresh=int(len(data_merge)*0.5))
dataclean = dataclean.fillna(0)

featurelist = list(dataclean.columns.values)
featurelist.remove('Id')
featurelist.remove('Response')
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(dataclean[featurelist], dataclean['Response'], test_size=0.1, random_state=42)

# param_grid = {
#               "criterion": ["gini", "entropy"],
#               "min_samples_split": [2,4],
#               "max_depth": [None,2,4],
#               "min_samples_leaf": [1,3,5],
#               "class_weight":["balanced","balanced_subsample"]
#               }

param_grid = {
              "criterion": ["gini", "entropy"],
              "min_samples_split": [2,4,5,6,7,8,9,10],
              "max_depth": [None,2,4],
              "min_samples_leaf": [1,3,5,6,7,8,9,10],
              "class_weight":["balanced","balanced_subsample"],
              'n_estimators':[10,20,30,40,50],
              'n_jobs':[-1]
              }

modeloptimal = grid_search.GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='f1', cv=5)
modeloptimal.fit(features_train, labels_train)

clf = modeloptimal.best_estimator_

#pred = clf.predict(features_test)
#accuracy = accuracy_score(labels_test, pred)


# test set 
test_numeric = pd.read_csv('test_numeric.csv')
test_date = pd.read_csv('test_date.csv')
data_merge = pd.merge(test_numeric, test_date, on='Id',suffixes=('tenum', 'tedate'))

makesubmit(clf,data_merge,featurelist,output="submit.csv")

print modeloptimal.best_estimator_ 