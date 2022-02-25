import pandas as pd 
import numpy as np 
pd.options.display.max_rows = 100

HousePriceDF = pd.read_csv("Ames_HousePrice.csv")


HousePriceDF=HousePriceDF.drop(['Alley','Fence','MiscFeature','PoolQC'],axis =1 )

HousePriceDF['LotFrontage'] = HousePriceDF['LotFrontage'].fillna(HousePriceDF.groupby('Neighborhood')['LotFrontage'].transform('mean'))



# Working on cleaning garage catagories: 

#garage sub DF for exmaning
garage = HousePriceDF.filter(regex='^Garage',axis=1)

garage.isna().sum() 


garage.loc[(garage["GarageType"].notnull()) & garage["GarageYrBlt"].isna()]

# want to look at rows 433 and 531

HousePriceDF['GarageType'].loc[433] = "None"

HousePriceDF['GarageCars'].loc[433] = 0.0

HousePriceDF['GarageArea'].loc[433] = 0.0


HousePriceDF["GarageYrBlt"].loc[531] = 1983

HousePriceDF["GarageFinish"].loc[531] = "Unf"

HousePriceDF["GarageQual"].loc[531] = "TA"

HousePriceDF["GarageCond"].loc[531] = "TA"

HousePriceDF.fillna('None', inplace=True)







