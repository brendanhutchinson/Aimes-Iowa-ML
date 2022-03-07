#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 18:02:10 2022

@author: mcmahon
"""
t
from manipulation2 import HousePriceDF        
        

# create a years since remodeled column   -  drop YearRemodAdd from DF  
HousePriceDF['YrSinceRm'] = HousePriceDF.YrSold - HousePriceDF.YearRemodAdd


# binary value for pool and misc feature - drop PoolArea & MiscVal from Df

HousePriceDf['Pool'] = HousePriceDF.PoolArea.apply(lambda x: 0 if x==0 else 1)

HousePriceDf['Misc'] = HousePriceDF.MiscVal.apply(lambda x: 0 if x==0 else 1)


# Finished basement square footage
HousePriceDF['BsmtFinTotSF'] = HousePriceDF.TotalBsmtSF - HousePriceCF.BsmtUnfSF


# Total Square Footage
HousePriceDF['TotalSF'] = (HousePriceDF.GrLivArea + 
                        HousePriceDf.TotalBsmtSF + 
                        HousePriceDF.GarageArea)

# Total Baths
HousePriceDF['TotalBath'] = ((HousePriceDF.FullBath + HousePriceDf.BsmtFullBath) +
                          0.5 * (HousePriceDf.HalfBath + HousePriceDf.BsmtHalfBath))


# lists and sublists of numeric and categorical variables 

numeric_all = ['GrLivArea', 'LotFrontage','LotArea', 'Street', 'LetShape', 'LandContour', 'Utilities', 
           'LotConfig', 'LandSlope', 'OverallQual','OverallCond' , 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
           'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'HeatingQC', 
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
           'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual'
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 
           'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal' , 'MoSold', 'YrSold', 
           'SaleCondition', 'YrSinceRm', 'Pool', 'Misc', 'BsmtFinTotSF', 'TotalSF', 'SchD_S']

categorical_all  = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType'
               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st' , 'Exterior2nd', 'MasVnrType'
               'Foundation', 'Heating', 'GarageType','SaleType' ]




# seperating dataframe for modeling 

y = HousePriceDF['SalePrice']

numeric = numeric_all 

categorical = categorical_all 


dummies = pd.get_dummies(HousePriceDF[categorical], drop_first=True)

X = pd.concat(HousePRiceDF[numeric], dummies)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

