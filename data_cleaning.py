import pandas as pd 
import numpy as np 
pd.options.display.max_rows = 100

HousePriceDF = pd.read_csv("Aimes-Iowa-ML\\Ames_HousePrice.csv")


# drop columns with over 98% missing values 
HousePriceDF=HousePriceDF.drop(['Alley','Fence','MiscFeature','PoolQC'],axis =1 )

# impute frontage based on mean for the neighborhood 
HousePriceDF['LotFrontage'] = HousePriceDF['LotFrontage'].fillna(HousePriceDF.groupby('Neighborhood')['LotFrontage'].transform('mean'))
HousePriceDF.loc[HousePriceDF['MSSubClass']==120, ['LotFrontage']].mean()
HousePriceDF.loc[HousePriceDF['MSSubClass']==160, ['LotFrontage']].mean()
HousePriceDF.iloc[1204,6] = 29.0
lotrow = [109,683]
HousePriceDF.loc[lotrow,['LotFrontage']]  = 46.0

# change FireplaceQu to 'None' for non existent fireplaces 
HousePriceDF.loc[HousePriceDF["Fireplaces"] == 0, "FireplaceQu"] = 'None'


# cleaning up the Basement columns 
HousePriceDF['BsmtQual'][HousePriceDF.Foundation == 'Slab'] = "None"
HousePriceDF['BsmtExposure'][HousePriceDF.Foundation == 'Slab'] = "None"
HousePriceDF['BsmtCond'][HousePriceDF.Foundation == 'Slab'] = "None"
HousePriceDF['BsmtFinType1'][HousePriceDF.Foundation == 'Slab'] = "None"
HousePriceDF['BsmtFinType2'][HousePriceDF.Foundation == 'Slab'] = "None"
HousePriceDF.loc[(HousePriceDF['BsmtFinSF1']== 0),32:36] = "None"
HousePriceDF.iloc[2434,37] = 'Unf'
HousePriceDF.iloc[912,32:36] = "None"
HousePriceDF.iloc[912,32:36] = "None"
HousePriceDF.loc[(HousePriceDF['BsmtFinSF1']== 0),'BsmtFinType2' ] = "None"
HousePriceDF.iloc[912,37] = 'None'
HousePriceDF.iloc[912,36] = 0
HousePriceDF.iloc[912,36] = 0
HousePriceDF.iloc[912,38:41] = 0 

HousePriceDF.iloc[912,48:50] = 0

HousePriceDF.iloc[2308,48:50] = 0

HousePriceDF.iloc[2442,44] = 'SBrkr'

HousePriceDF.iloc[358,27:29] = 'None'
HousePriceDF.iloc[1355,27:29] = 'None'

# MasVnrType/Area cleaning 
HousePriceDF.loc[(HousePriceDF['Exterior2nd']== 'VinylSd' )&(HousePriceDF.MasVnrType.isna()==True), ['MasVnrType' ]] = 'None'
HousePriceDF.loc[(HousePriceDF['Exterior2nd']== 'VinylSd' )&(HousePriceDF.MasVnrType.isna()==True), ['MasVnrArea' ]] = 0
# Electrical cleaning
HousePriceDF.iloc[2442,44] = 'SBrkr'




# cleaning up the Garage columns 
HousePriceDF['GarageType'].iloc[433] = "None"

HousePriceDF['GarageCars'].iloc[433] = 0.0 	#maintain float

HousePriceDF['GarageArea'].iloc[433] = 0.0  #maintain float 

HousePriceDF["GarageYrBlt"].iloc[531] = 1983

HousePriceDF["GarageFinish"].iloc[531] = "Unf"

HousePriceDF["GarageQual"].iloc[531] = "TA"

HousePriceDF["GarageCond"].iloc[531] = "TA"


# replace emtpy GarageYrBlt with YearBuilt for missing garages to maintin float 
HousePriceDF['GarageYrBlt'].fillna(HousePriceDF['YearBuilt'], inplace=True)


# DO NOT UNCOMMENT THIS UNTIL SURE WE ARE DONE IMPUTING 
HousePriceDF.fillna('None', inplace=True)




