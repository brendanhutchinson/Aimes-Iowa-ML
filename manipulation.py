#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:11:15 2022

@author: mcmahon / Brendan/ Francesco / Yuni 
"""

#import cleaning.py
import numpy as np
import pandas as pd

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

## fireplace encoding : 0 ,1 ,2 = 2 + 
FireplacesDict = {0:0,1:1,2:2,3:2,4:2}
HousePriceDF.replace({"Fireplaces": FireplacesDict}, inplace=True)


# roof material 
roofdict = {'WdShake': 'wood','WdShngl' :'wood','Metal':'other','Roll':'other','Membran':'other'}
HousePriceDF.replace({"RoofMatl": roofdict},inplace=True)

## heating 
heatingdict = {'Grav':'other','Wall':'other','OthW':'other','Floor':'other'}
HousePriceDF.replace({"Heating": heatingdict},inplace=True)



# functional 
funct_dict = {'Typ':0, 'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5,'Sal':6}
HousePriceDF.replace({"Functional": funct_dict},inplace=True)

## foundation
found_dict = {'Stone':'other','Wood':'other'}
HousePriceDF.replace({"Foundation": found_dict},inplace=True)

## Sale type 
SaleTypeDict = {'New':'New','COD':'COD','ConLD':'Con','CWD':'WD','ConLI':'Con','Con':'Con','Oth':'Con','VWD':'WD','ConLw':'Con','WD ':'WD'}
HousePriceDF.replace({"SaleType": SaleTypeDict},inplace = True)

# Condition 1
Condition1Dict = {'Norm':'Norm','Feedr':'Feedr','Artery':'Artery','RRAn':'RR','PosN':'Pos','RRAe':'RR','PosA':'Pos','RRNn':'RR','RRNe':'RR'}
HousePriceDF.replace({"Condition1": Condition1Dict},inplace = True)

# Condition 2
Condition2Dict = {'Norm':'Norm','Feedr':'Feedr','PosN':'Pos','Artery':'Artery','PosA':'Pos','RRNn':'RR','RRAn':'RR','RRAe':'RR'}
HousePriceDF.replace({"Condition2": Condition2Dict},inplace = True)


## MSSUB class int - string 
HousePriceDF['MSSubClass'] = HousePriceDF['MSSubClass'].astype('str')

## MSZoning
MSZoningDict = {'RL':'RL', 'C (all)':'Other', 'RM':'RM', 'FV':'FV', 'RH':'RH', 'I (all)':'Other', 'A (agr)':'Other'}
HousePriceDF.replace({"MSZoning":MSZoningDict},inplace = True)

## LotConfig
HousePriceDF['LotConfig'] = np.where((HousePriceDF['LotConfig'] == "FR2") | (HousePriceDF['LotConfig'] == "FR3"), "FR", HousePriceDF['LotConfig'])

# combine values in Exterior columns

HousePriceDF.Exterior1st = HousePriceDF.Exterior1st.map({
    'VinylSd' : 'VinylSd', 
    'HdBoard' : 'CompBoard', 
    'MetalSd' : 'MetalSd', 
    'Wd Sdng' : 'Wood', 
    'Plywood' : 'Wood', 
    'WdShing' : 'Wood',
    'CemntBd' : 'CompBoard', 
    'BrkComm' : 'Brick', 
    'BrkFace' : 'Brick', 
    'Stucco' : 'Cement', 
    'ImStucc' : 'Cement', 
    'PreCast' : 'Cement', 
    'CBlock' : 'Cement', 
    'AsphShn' :'Other', 
    'AsbShng' : 'Other'})

HousePriceDF.Exterior2nd = HousePriceDF.Exterior2nd.map({
    'VinylSd' : 'VinylSd', 
    'HdBoard' : "CompBoard", 
    'MetalSd' : 'MetalSd', 
    'Wd Sdng' : 'Wood', 
    'Plywood' : 'Wood', 
    'Wd Shng' : 'Wood',
    'CmentBd' : 'CompBoard', 
    'Brk Cmn' : 'Brick', 
    'BrkFace' : 'Brick', 
    'Stucco' : 'Cement', 
    'ImStucc' : 'Cement', 
    'PreCast' : 'Cement', 
    'CBlock' : 'Cement', 
    'AsphShn' :'Other', 
    'AsbShng' : 'Other',
    'Stone' : 'Other'})









