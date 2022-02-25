import pandas as pd 
import numpy as np 
pd.options.display.max_rows = 100

HousePriceDF = pd.read_csv("Aimes-Iowa-ML\\Ames_HousePrice.csv")


# drop columns with over 98% missing values 
HousePriceDF=HousePriceDF.drop(['Alley','Fence','MiscFeature','PoolQC'],axis =1 )

# impute frontage based on mean for the neighborhood 
HousePriceDF['LotFrontage'] = HousePriceDF['LotFrontage'].fillna(HousePriceDF.groupby('Neighborhood')['LotFrontage'].transform('mean'))


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


# cleaning up the Garage columns 
HousePriceDF['GarageType'].loc[433] = "None"

HousePriceDF["GarageYrBlt"].loc[531] = 1983

HousePriceDF["GarageFinish"].loc[531] = "Unf"

HousePriceDF["GarageQual"].loc[531] = "TA"

HousePriceDF["GarageCond"].loc[531] = "TA"



# DO NOT UNCOMMENT THIS UNTIL SURE WE ARE DONE IMPUTING 
#HousePriceDF.fillna('None', inplace=True)



