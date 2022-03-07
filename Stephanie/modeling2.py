#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import outputs from cleaning and manipulation modules
from manipulation2 import *


# import libraries and functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler 




# set df = whatever the final output of our dataframe is 

df = HousePriceDF


# lists and sublists of numeric and categorical variables 

numeric_all = ['GrLivArea', 'LotFrontage','LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 
            'LandSlope', 'OverallQual','OverallCond' , 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
           'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'HeatingQC', 
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
           'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 
           'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal' , 'MoSold', 'YrSold', 
           'SaleCondition', 'YrSinceRm', 'Pool', 'Misc', 'BsmtFinTotSF', 'TotalSF']

categorical_all  = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st' , 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'GarageType','SaleType' , 'LotConfig']




# seperating dataframe for modeling 

y = df['SalePrice']


# input which subset of numeric & categorical variables to be used 
numeric = numeric_all 

categorical = categorical_all 


# DO NOT TOUCH THIS CODE
dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# initiate and fit a full linear model 

lm = LinearRegression()
lm.fit(X_train, y_train)
print('The Linear R^2 is equal to %.3f' %(lm.score(X_test, y_test)))
#print('The Linear intercept is %.3f' %(lm.intercept_))
#print('The Linear slopes are %s' %(lm.coef_))

# initiate Ridge

rdg = Ridge()
rdg.fit(X_train, y_train)
print('The Ridge R^2 is equal to %.3f' %(rdg.score(X_test, y_test)))
#print('The Ridge intercept is %.3f' %(rdg.intercept_))
#print('The Ridge slopes are %s' %(rdg.coef_))

# initiate Lasso 

ls = Lasso()
ls.fit(X_train, y_train)
print('The Lasso R^2 is equal to %.3f' %(ls.score(X_test, y_test)))
#print('The Lasso intercept is %.3f' %(?))
#print('The Lasso slopes are %s' %(?))
print('The Lasso number of features are %s' %(ls.n_features_in_))
print('The Lasso featurs are %s' %(ls.feature_names_in_))

# initiate ElasticNet

alpha = 1.0
l1_ratio = 0.5
eln = ElasticNet()
eln.set_params(alpha=alpha, l1_ratio= l1_ratio)
eln.fit(X_train, y_train)
print('The ElasticNet R^2 is equal to %.3f' %(eln.score(X_test, y_test)))
#print('The ElasticNet intercept is %.3f' %(?))
#print('The ElasticNet slopes are %s' %(?))

