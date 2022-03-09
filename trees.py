#!/usr/bin/env python3

from random import random
from manipulation import * 
from modeling import *
from manipulation import * 
from sklearn import ensemble
import time
time
# Selected features 
numeric_1 = ['GrLivArea', 'LotFrontage','LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 
            'LandSlope', 'OverallQual','OverallCond' , 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'HeatingQC', 
             'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
           'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageFinish', 
           'GarageCars', 'GarageArea', 'GarageQual',  'PavedDrive', 'WoodDeckSF', 
           'EnclosedPorch',  'ScreenPorch', 'PoolArea', 'MiscVal' , 'MoSold', 'YrSold', 
           'SaleCondition', 'YrSinceRm','Pool', 'Misc', 'BsmtFinTotSF', 'IAstateDist', 'AirportDist', 'TotalBath']

categorical_1  = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st' , 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'GarageType','SaleType' , 'LotConfig' ] 



# Random Forest 

randomForest = ensemble.RandomForestClassifier()
bagging = ensemble.BaggingClassifier()




# input which subset of numeric & categorical variables to be used 


numeric = numeric_1

categorical = categorical_1


# DO NOT TOUCH THIS CODE 
dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)

# if subsetting use the code below before running tts
# X = X[subset]

#  split data for training and testing 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)





from sklearn import ensemble
randomForest = ensemble.RandomForestRegressor()
bagging = ensemble.BaggingRegressor()
# Random Forest Default parameters 


randomForest.fit(X_train,y_train)
randomForest.score(X_train,y_train)
randomForest.score(X_test,y_test)

# baggin default parameters 

bagging.fit(X_train, y_train)
bagging.score(X_train,y_train)
bagging.score(X_test,y_test)


# bagging

bagging.set_params(n_estimators=50, random_state=42, max_features=20)
bagging.fit(X_train, y_train)
print("The training error is: %.5f" %(1 - bagging.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - bagging.score(X_test, y_test)))
print("The training accuracy is: %.5f" %( bagging.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(bagging.score(X_test, y_test)))






# Random Forest

randomForest.set_params(random_state=42, n_estimators=100, max_features=20)
randomForest.fit(X_train,y_train) 
randomForest.score(X_train,y_train)
randomForest.score(X_test,y_test)

randomForest.set_params(random_state=42, n_estimators=100, max_features=5)
randomForest.fit(X_train,y_train) 
randomForest.score(X_train,y_train)
randomForest.score(X_test,y_test)


randomForest.set_params(n_estimators=50, random_state=42, max_features=10,oob_score=True)
randomForest.fit(X_train, y_train)
print("The training error  is: %.5f" %(1 - randomForest.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - randomForest.score(X_test, y_test)))
print("The training accuracy is: %.5f" %(randomForest.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(randomForest.score(X_test, y_test)))



# Grid Search 


randomForest.set_params(oob_score=True)
grid_tree = {
    'max_depth': range(1, 30),
    'n_estimators': range(10, 110, 10),
    'min_samples_split': range(2,10,2)
    
}

randomForest.set_params(random_state=0)
grid_search_tree = GridSearchCV(randomForest,grid_tree, cv=5, n_jobs=-1, return_train_score=True)
grid_search_tree.fit(X_train, y_train)


print('Best parameters: '+ str(grid_search_tree.best_params_))
print(grid_search_tree.best_score_)
print(randomForest.oob_score_)



grid_tree2 = {
    'max_depth': range(1, 20),
    'n_estimators': range(10, 1000, 50),
    'max_features' : [10,30,50,70],
}



randomForest.set_params(random_state=42)
grid_search_tree = ms.GridSearchCV(randomForest,grid_tree2, cv=5, n_jobs=-1)
 

grid_search_tree.fit(X_train, y_train)


print('Best parameters: '+ str(grid_search_tree.best_params_))
print(grid_search_tree.best_score_)

# Scoring test data set with tuned forest parameters 

randomForest.set_params(n_estimators=50, random_state=42, max_features=50 ,oob_score=True)
randomForest.fit(X_train, y_train)
print("The training error  is: %.5f" %(1 - randomForest.score(X_train, y_train)))
print("The test error is: %.5f" %(1 - randomForest.score(X_test, y_test)))
print("The training accuracy is: %.5f" %(randomForest.score(X_train, y_train)))
print("The test accuracy is: %.5f" %(randomForest.score(X_test, y_test)))



randomForest.set_params(random_state=42, n_estimators=100, max_features=70)
randomForest.fit(X_train,y_train) # fit 
randomForest.score(X_train,y_train)
print(randomForest.score(X_test,y_test))

randomForest.set_params(random_state=42, n_estimators=100, max_features=30, max_depth = 14)
randomForest.fit(X_train,y_train) 
randomForest.score(X_train,y_train)
print(randomForest.score(X_test,y_test))

# Random Forest Best Score 

randomForest.set_params(random_state=42, n_estimators=700, max_features=30, max_depth = 14)
randomForest.fit(X_train,y_train) 
print(randomForest.score(X_train,y_train))
print(randomForest.score(X_test,y_test))
print("The best  test accuracy is: %.5f" %(randomForest.score(X_test, y_test)))



# sorting feature importances 

tree_final = grid_search_tree.best_estimator_
feature_importance = list(zip(X_train.columns, tree_final.feature_importances_))
dtype = [('feature', 'S10'), ('score', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
sort_X = np.sort(feature_importance, order='score')[::-1]
name, score = zip(*list(sort_X))
pd.DataFrame({'feature':name,'score':score}).plot.barh(x='name', y='score')


















