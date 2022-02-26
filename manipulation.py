#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:11:15 2022

@author: mcmahon
"""

#import cleaning.py

# Change to boolean columns

HousePriceDF.Street = HousePriceDF.Street.map({'Pave': 1, 'Grvl':0})

HousePriceDF.LotShape = HousePriceDF.LotShape.apply(lambda x: 1 if x == 'Reg' else 0)

HousePriceDF.LandContour = HousePriceDF.LandContour.apply(lambda x: 1 if x == 'Lvl' else 0)

HousePriceDF.Utilities = HousePriceDF.Utilities.apply(lambda x: 1 if x == 'AllPub' else 0)

HousePriceDF.LandSlope = HousePriceDF.LandSlope.apply(lambda x: 1 if x == 'Gtl' else 0)

HousePriceDF.CentralAir = HousePriceDF.CentralAir.map({'Y': 1, 'N':0})

HousePriceDF.Electrical = HousePriceDF.Electrical.apply(lambda x: 1 if x == 'SBrkr' else 0)

HousePriceDF.PavedDrive = HousePriceDF.PavedDrive.apply(lambda x: 1 if x == 'Y' else 0)

HousePriceDF.SaleCondition = HousePriceDF.SaleCondition.apply(lambda x: 1 if x == 'Normal' else 0)



# Encoding Bsmt columns 
BsmtCondDict = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
HousePriceDF.replace({"BsmtCond": BsmtCondDict},inplace=True)
HousePriceDF.replace({"BsmtQual": BsmtCondDict},inplace=True)

BsmtFinType = {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
HousePriceDF.replace({"BsmtFinType1": BsmtFinType,'BsmtFinType2':BsmtFinType},inplace=True)

BsmtExpDict = {'None':0,'No': 1,'Mn':2,'Av':3,'Gd':4}
HousePriceDF.replace({"BsmtExposure": BsmtExpDict},inplace=True)

#Encoding Kitchen Quality 

QualDict = {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
HousePriceDF.replace({"KitchenQual": QualDict},inplace=True)

# Encoding Heating QC 

HousePriceDF.replace({"HeatingQC": QualDict},inplace=True)

# Encoding Garage Qual/Condition

GarageQDict = {'None': 0,'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
HousePriceDF.replace({"GarageQual": GarageQDict, 'GarageCond':GarageQDict},inplace=True)
GarageFinDict = {'None':0, 'Unf':1,'RFn':2,"Fin":3}
HousePriceDF.replace({"GarageFinish": GarageFinDict},inplace=True)

# Encoding Exterior Quality 

HousePriceDF.replace({"ExterQual": QualDict,'ExterCond':QualDict},inplace=True)

# Encoding Fireplace columns 


HousePriceDF.replace({"FireplaceQu": GarageQDict},inplace=True)


