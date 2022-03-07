#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:59:32 2022

@author: mcmahon
"""

from manipulation import * 


from sklearn import ensemble
randomForest = ensemble.RandomForestClassifier()
bagging = ensemble.BaggingClassifier()
# bagging
bagging.set_params(n_estimators=50, random_state=42, max_features=57)
bagging.fit(X_train, y_train)
print("The training error of bagging forest is: %.5f" %(1 - bagging.score(X_train, y_train)))
print("The test     error of bagging forest is: %.5f" %(1 - bagging.score(X_test, y_test)))
np.mean([estimator.score(X_test, y_test=="spam") for estimator in bagging])

# Random Forest
randomForest.set_params(n_estimators=50, random_state=42, max_features=10)
randomForest.fit(X_train, y_train)
print("The training error of random forest is: %.5f" %(1 - randomForest.score(X_train, y_train)))
print("The test     error of random forest is: %.5f" %(1 - randomForest.score(X_test, y_test)))
np.mean([estimator.score(X_test, y_test=="spam") for estimator in randomForest])