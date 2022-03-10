#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:05:16 2022

@author: mcmahon
"""

from manipulation2 import *


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score 


import xgboost as xgb



totaldf = pd.read_csv('finaldf.csv')

# set df = whatever the final output of our dataframe is 

df = HousePriceDF



# features 

num_all = ['GrLivArea', 'LotFrontage','LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 
            'LandSlope', 'OverallQual','OverallCond' , 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
           'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'HeatingQC', 
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
           'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 
           'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal' , 'MoSold', 'YrSold', 
           'SaleCondition', 'YrSinceRm', 'Pool', 'Misc', 'BsmtFinTotSF', 'TotalSF', 'TotalBath']

cat_all  = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st' , 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'GarageType','SaleType' , 'LotConfig']



num_wins = [
    'YearBuilt', 'YrSinceRm', 'BsmtUnfSF' , 'WoodDeckSF', '2ndFlrSF', 
    'ScreenPorch', 'GrLivArea', 'BsmtFinSF1', 'MasVnrArea', 'LotArea', 'Pool',
    'KitchenQual', 'BsmtFullBath', 'BedroomAbvGr', 'LotFrontage',
    'Utilities','FullBath','Fireplaces', 'FireplaceQu','TotalBsmtSF',
    'GarageArea', 'HalfBath', 'ExterQual','GarageCars','1stFlrSF','OverallQual'
]

cat_wins = [
            'Neighborhood', 'MSSubClass', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'HouseStyle', 'GarageType'
]



# input which subset of numeric & categorical variables to be used 
numeric = num_all 

categorical = cat_all


# DO NOT TOUCH THIS CODE

#to remove outliers
df = df.loc[df['SalePrice']<475000]

y = np.log(df['SalePrice'])

dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# instantiate xgboost regressor
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', 
                          n_estimators=100, max_depth=30, 
                          booster = 'gbtree', 
                          random_state = 42, 
                          learning_rate = 0.05, 
                          subsample=0.8)




# train and fit with xgboost built in validation 

DM_train = xgb.DMatrix(X_train, label=y_train)
DM_test = xgb.DMatrix(X_test, label=y_test)


# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", 
          "max_depth":5, 'learning_rate': 0.1, 
          'colsample_bytree' : 0.3}

xg_reg = xgb.train(params = params, 
                   dtrain = DM_train, 
                   num_boost_round=100
                   )

pred = xg_reg.predict(DM_test)

r2_score(y_test, pred) #0.914477145112945

# take a look at feature importance 
xgb.plot_importance(xg_reg)



# tune parameters using GridSearchCV 

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    "objective":["reg:squarederror"], 
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7],
    'n_estimators': [50, 100, 500],
    'max_depth': [5, 10], 
    'sub_sample' : [0.7, 0.8, 0.9]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator = gbm, 
param_grid = gbm_param_grid,
scoring = "r2", 
cv = 4,
verbose = 1, 
n_jobs=-1)


# Fit grid_mse to the data
grid_mse.fit(X_train, y_train)

# Print the best parameters the corresponding r2 
print("Best parameters found: ", grid_mse.best_params_)
print("Best train R2 found: ", 
      grid_mse.best_estimator_.score(X_train, y_train))
print("Best test R2 found: ", 
      grid_mse.best_estimator_.score(X_test, y_test))



# =============================================================================
# 
# best params: {'colsample_bytree': 0.3,
#  'learning_rate': 0.05,
#  'max_depth': 5,
#  'n_estimators': 500,
#  'objective': 'reg:squarederror',
#  'sub_sample': 0.7}



# subset with outliers still included
# best r2_train : 0.9851760032987463
# best r2_test : 0.9197790991254833


# subset after removing outliers
# Best train R2 found:  0.9851161725811441
# Best test R2 found:  0.9275614826543322


# all features, outliers removed - colsample_bytree : 0.5
#Best train R2 found:  0.9904128309960637
#Best test R2 found:  0.9454579871546015


# =============================================================================





#special test 

new_df = pd.concat([X_test, y_test], axis = 1)
new_df = new_df.loc[new_df['SalePrice']<np.log(475000)]


new_ytst = new_df['SalePrice']
new_xtst = new_df.drop('SalePrice', axis=1)

 grid_mse.best_estimator_.score(new_xtst, new_ytst)
 




xgb.plot_importance(grid_mse.best_estimator_)

