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

