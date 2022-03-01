#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 18:02:10 2022

@author: mcmahon
"""

from manipulation2 import HousePriceDF        
        

# create a years since remodeled column   -  drop YearRemodAdd from DF  
HousePriceDF.YrSinceRm = HousePriceDF.YrSold - HousePriceDf.YearRemodAdd


# binary value for pool and misc feature - drop PoolArea & MiscVal from Df

HousePriceDf.Pool = HousePriceDF.PoolArea.apply(lambda x: 0 if x==0 else 1)

HousePriceDf.Misc = HousePriceDF.MiscVal.apply(lambda x: 0 if x==0 else 1)


# Finished basement square footage
HousePriceDF.BsmtFinTotSF = HousePriceDF.TotalBsmtSF - HousePriceCF.BsmtUnfSF


# Total Square Footage
HousePriceDF.TotalSF = (HousePriceDF.GrLivArea + 
                        HousePriceDf.TotalBsmtSF + 
                        HousePriceDF.GarageArea)

# Total Baths
HousePriceDF.TotalBath = ((HousePriceDF.FullBath + HousePriceDf.BsmtFullBath) +
                          0.5 * (HousePriceDf.HalfBath + HousePriceDf.BsmtHalfBath))

