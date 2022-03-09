

from trees import *
from catboost import CatBoostRegressor
import pandas as pd 
import numpy as np 



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



# dummifying variables 

numeric = numeric_1

categorical = categorical_1

y = np.log(df['SalePrice'])

# DO NOT TOUCH THIS CODE 
dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Cat Boost with defualt parameters 

model = CatBoostRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.score(X_test,y_test)


# Cat Boost Grid Search 

parameters = {
              'learning_rate': np.arange(.01,1),
              'max_depth': range(1,25),
               'n_estimators': [10,20,40,50,80,100,200,300]}
                 


grid_cat1 = GridSearchCV(model, param_grid= parameters, cv=5, n_jobs=-1).fit(X_train,y_train)

parameters2 = {
              'learning_rate': [.001,.01,.1,.5,1],
              'max_depth': [3,4,6,10],
               'n_estimators': [100,200,300,500,700,1000]}
                 
grid_cat2 = GridSearchCV(model, param_grid= parameters2, cv=5, n_jobs=-1).fit(X_train,y_train)


catboost = CatBoostRegressor()
parameters2 = {
              'learning_rate': [.08,.1,.15],
              'max_depth': [4,6,7],
               'n_estimators': [500,600,700,900]}
                 


grid_cat = GridSearchCV(catboost, param_grid= parameters2, cv=3, n_jobs=-1).fit(X_train,y_train)





print('optimal grid parameters: '+ str(grid_cat.best_params_))
print(grid_cat.best_score_)

print('optimal grid parameters: '+ str(grid_cat.best_params_))
print(grid_cat1.best_score_)

print('optimal grid parameters: '+ str(grid_cat.best_params_))
print(grid_cat2.best_score_)


# Scoring Test data set with tuned parameters 



cat = CatBoostRegressor()
cat.set_params(random_state = 42 ,learning_rate = .08, max_depth = 4 , n_estimators = 900)
cat.fit(X_train,y_train)
cat.score(X_test,y_test)




cat.set_params(random_state = 42 ,learning_rate = 0.1, max_depth = 4 , n_estimators = 500)
cat.fit(X_train,y_train)
cat.score(X_test,y_test)

cat.set_params(random_state = 42 ,learning_rate = 0.1, max_depth = 6 , n_estimators = 700)
cat.fit(X_train,y_train)
cat.score(X_test,y_test)  


# Best Cat boost score 


cat.set_params(random_state = 42 ,learning_rate = 0.1, max_depth = 4 , n_estimators = 900)
cat.fit(X_train,y_train)
cat.score(X_test,y_test)               


# Feature Importances 

tree_best = grid_cat.best_estimator_
feature_importance = list(zip(X_train.columns, tree_best.feature_importances_))
dtype = [('feature', 'S10'), ('score', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
sort_X = np.sort(feature_importance, order='score')[::-1]
features, score = zip(*list(sort_X))
featuredf = pd.DataFrame({'features':features,'score':score})



featuredf.features =[str(i) for i in featuredf.features]
featuredf.features =[i.replace('b','') for i in featuredf.features]
featuredf.features =[i.replace('\'','') for i in featuredf.features]



# scatterplot of Predicted vs Test target
import math 
y_pred = model.predict(X_test)
y_test =[math.exp(i) for i in y_test]
y_pred = [math.exp(i) for i in y_pred]
px.scatter(x=y_test,y=y_pred,labels={'x':'Test Target ', 'y':'Model Predictions'})






### df with removed outliers 
 
outlierdf =df.loc[df['SalePrice']< 475000] 
# input which subset of numeric & categorical variables to be used 
numeric = numeric_1

categorical = categorical_1

y1 = np.log(outlierdf['SalePrice'])
# DO NOT TOUCH THIS CODE 
dummies = pd.get_dummies(outlierdf[categorical], drop_first=True)

X1 = pd.concat([outlierdf[numeric], dummies], axis = 1)

# if subsetting use the code below before running tts
# X = X[subset]



X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30, random_state=42)



# cat boost with outliers removed
cat1 = CatBoostRegressor()
cat1.set_params(random_state = 42 ,learning_rate = .1, max_depth = 6 , n_estimators = 500)
cat1.fit(X_train1,y_train1)
cat1.score(X_test1,y_test1)

# cat boost with outliers removed 

cat1 = CatBoostRegressor()
cat1.set_params(random_state = 42 ,learning_rate = .1, max_depth = 6 , n_estimators = 700)
cat1.fit(X_train1,y_train1)
cat1.score(X_test1,y_test1)



## gradient boosting regressor 

from sklearn.ensemble import  GradientBoostingRegressor


gbm = GradientBoostingRegressor()

gbm.fit(X_train,y_train)

gbm.score(X_train,y_train)

gbm.score(X_test,y_test)

#gradient boosting regression grid search 


parameters3 = {
              'learning_rate': [.08,.09,.1,.12],
              'max_depth': [3,4,6,7,8],
               'n_estimators': [500,550,600,700,900]}
                 


grid_gbm = GridSearchCV(gbm, param_grid= parameters3, cv=3, n_jobs=-1).fit(X_train,y_train)

print('Best parameters: '+ str(grid_gbm.best_params_))
print(grid_gbm.best_score_)



# testing best parameters 

gbm.set_params(learning_rate = .08, max_depth =6 , n_estimators = 700)
gbm.fit(X_train,y_train)
gbm.score(X_test,y_test)

# gradient boosting best score 

gbm.set_params(learning_rate = .08, max_depth =3 , n_estimators = 550)
gbm.fit(X_train,y_train)
gbm.score(X_test,y_test)

