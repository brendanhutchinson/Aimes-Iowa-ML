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


totaldf = pd.read_csv('finaldf.csv')

# set df = whatever the final output of our dataframe is 

df = totaldf


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
           'SaleCondition', 'YrSinceRm', 'Pool', 'Misc', 'BsmtFinTotSF', 'TotalSF', 'IAstateDist', 'AirportDist', 'TotalBath']

categorical_all  = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st' , 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'GarageType','SaleType' , 'LotConfig' ,'SchD_S']



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

# note the winners of the first round of Lasso were all numeric 
# set X = df[lasso1]
lasso1 = ['YearBuilt', 'YrSinceRm', 'TotalSF', 'BsmtUnfSF' , 'WoodDeckSF', 'GarageArea', '2ndFlrSF', 
          'ScreenPorch', 'GrLivArea', 'BsmtFinSF1', 'MasVnrArea', 'LotArea']

lasso2 =['Condition2_Pos', 'Neighborhood_GrnHill', 'Neighborhood_StoneBr', 'Pool', 'Exterior1st_VinylSd', 'Utilities',
         'MSSubClass_150', 'RoofMatl_wood', 'MSSubClass_50', 'GarageType_None', 'MSSubClass_75', 'SaleType_New', 
         'Neighborhood_NridgHt', 'MSSubClass_30', 'Exterior1st_Wood', 'Neighborhood_NPkVill', 'MSSubClass_20', 
         'HouseStyle_SLvl', 'MSSubClass_40', 'Exterior2nd_VinylSd']


lasso_combo = lasso1 + lasso2



num_2 = ['YearBuilt', 'YrSinceRm', 'TotalSF', 'BsmtUnfSF' , 'WoodDeckSF', 'GarageArea', '2ndFlrSF', 
          'ScreenPorch', 'GrLivArea', 'BsmtFinSF1', 'MasVnrArea', 'LotArea', 'Pool', 'Utilities']

cat_2 = ['Neighborhood', 'MSSubClass', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 
         'Exterior2nd', 'HouseStyle', 'GarageType']





num_wins = [
    'YearBuilt', 'YrSinceRm', 'BsmtUnfSF' , 'WoodDeckSF', '2ndFlrSF', 
    'ScreenPorch', 'GrLivArea', 'BsmtFinSF1', 'MasVnrArea', 'LotArea', 
    'Pool','KitchenQual', 'BsmtFullBath', 'BedroomAbvGr', 'LotFrontage',
    'Utilities','FullBath','Fireplaces', 'FireplaceQu','TotalBsmtSF',
    'GarageArea', 'HalfBath', 'ExterQual','GarageCars','1stFlrSF','OverallQual'
]

cat_wins = [
            'Neighborhood', 'MSSubClass', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'HouseStyle', 'GarageType'
]


# input which subset of numeric & categorical variables to be used 
numeric = num_wins

categorical = cat_wins


# DO NOT TOUCH THIS CODE

# remove outliers
df = df.loc[df['SalePrice']<475000]

y = np.log(df['SalePrice'])
# y2 = df['SalePrice']

dummies = pd.get_dummies(df[categorical], drop_first=True)

X = pd.concat([df[numeric], dummies], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# only use this for interpreting coefficients
# y2 = df['SalePrice']
# X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)


# initiate and fit a full linear model 

lm = LinearRegression()
lm.fit(X_train, y_train)
print('The Linear R^2 is equal to %.3f' %(lm.score(X_test, y_test)))
#print('The Linear intercept is %.3f' %(lm.intercept_))
#print('The Linear slopes are %s' %(lm.coef_))


# talk about ceofs and their interpretation
coefs = lm.coef_

features = pd.Series(coefs, index = X.columns)
features = features.sort_values(ascending=False)

features_df = pd.DataFrame(features)
features_df.rename(columns={0:'coef'}, inplace = True)


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


coefs = abs(ls.coef_)

features = pd.Series(coefs, index = X.columns)
features.sort_values(ascending=False).head(20)

# initiate ElasticNet

alpha = 1.0
l1_ratio = 0.5
eln = ElasticNet()
eln.set_params(alpha=alpha, l1_ratio= l1_ratio)
eln.fit(X_train, y_train)
print('The ElasticNet R^2 is equal to %.3f' %(eln.score(X_test, y_test)))
#print('The ElasticNet intercept is %.3f' %(?))
#print('The ElasticNet slopes are %s' %(?))



# GridSearch for Lasso optimization 


pipe = Pipeline([
                     ('scaler', StandardScaler()),
                     ('model', Lasso())
])


gs = GridSearchCV(pipe,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="r2",verbose=3
                      )


gs.fit(X_train,y_train)
print('The Lasso GS R^2 is equal to %.3f' %(gs.score(X_test, y_test))) 
print('The Lasso GS alpha was equal to %.3f' %(gs.best_params_))




# using GridSearch for ElasticNet

pipe2 = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model', ElasticNet())
])

gs2 = GridSearchCV(pipe2,
                      {'model__alpha':np.arange(0.1,10,0.1), 'model__l1_ratio' : np.arange(0.1,1, 10)},
                      cv = 5, scoring="r2",verbose=3
                      )

gs2.fit(X_train, y_train)
print('The ElasticNet GS R^2 is equal to %.3f' %(gs2.score(X_test, y_test))) 
print('The ElasticNet GS alpha and rho values were equal to %.3f' %(gs2.best_params_ ))

 

# Gridsearch for Ridge 

pipe = Pipeline([
                     ('scaler', StandardScaler()),
                     ('model', Ridge())
])


gs = GridSearchCV(pipe,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="r2",verbose=3
                      )


gs.fit(X_train,y_train)
print('The Ridge GS R^2 is equal to %.3f' %(gs.score(X_test, y_test))) 
print('The Ridge GS alpha was equal to %.3f' %(gs.best_params_))


