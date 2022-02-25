import pandas as pd 
import numpy as np 
pd.options.display.max_rows = 100

HousePriceDF = pd.read_csv("Aimes-Iowa-ML\\Ames_HousePrice.csv")


HousePriceDF=HousePriceDF.drop(['Alley','Fence','MiscFeature','PoolQC'],axis =1 )

HousePriceDF['LotFrontage'] = HousePriceDF['LotFrontage'].fillna(HousePriceDF.groupby('Neighborhood')['LotFrontage'].transform('mean'))

HousePriceDF.loc[HousePriceDF["Fireplaces"] == 0, "FireplaceQu"] = 'None'


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



