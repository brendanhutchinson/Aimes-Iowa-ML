

# Running cleaning and feature engineering and merging


mapdata = pd.read_csv('Ames Real Estate Data.csv')

run data_cleaning.py

run manipulation
from data_cleaning import *
from manipulation import *

locationdf = mapdata[['SchD_S', 'Prop_Addr', 'GeoRefNo' ,'MapRefNo']]

HousePriceDF.merge(locationdf, left_on = 'PID', right_on = 'MapRefNo',how = 'left').size

DF = HousePriceDF.merge(locationdf, left_on = 'PID', right_on = 'MapRefNo',how = 'inner')

# formatting addresses for Ames IA 

DF['Prop_Addr'] = DF['Prop_Addr'].apply(lambda x: "{}{}".format(x, ' Ames, IA'))


# Getting Longitude and Latitude with googlemaps API 
import googlemaps
import time

gmaps = googlemaps.Client(key=apikey)

DF['latitude']=''
DF['longitude']=''


for i in DF.Prop_Addr[:400]: 
    time.sleep(1)  
    geocode_result = gmaps.geocode(i)
    idx = DF['Prop_Addr'].isin([i])
    DF.loc[idx,"latitude"]= geocode_result[0]['geometry']['location']['lat']
    DF.loc[idx,"longitude"] = geocode_result[0]['geometry']['location']['lng']
    
for i in DF.Prop_Addr[400:1000]: 
    time.sleep(1)
    geocode_result = gmaps.geocode(i)
    idx = DF['Prop_Addr'].isin([i])
    DF.loc[idx,"latitude"]= geocode_result[0]['geometry']['location']['lat']
    DF.loc[idx,"longitude"] = geocode_result[0]['geometry']['location']['lng']

for i in DF.Prop_Addr[1000:1700]:
    time.sleep(1)
    geocode_result = gmaps.geocode(i)
    idx = DF['Prop_Addr'].isin([i])
    DF.loc[idx,"latitude"]= geocode_result[0]['geometry']['location']['lat']
    DF.loc[idx,"longitude"] = geocode_result[0]['geometry']['location']['lng']


for i in DF.Prop_Addr[1700:2400]:
    time.sleep(1)
    geocode_result = gmaps.geocode(i)
    idx = DF['Prop_Addr'].isin([i])
    DF.loc[idx,"latitude"]= geocode_result[0]['geometry']['location']['lat']
    DF.loc[idx,"longitude"] = geocode_result[0]['geometry']['location']['lng']

for i in DF.Prop_Addr[2400:2603]: 
    time.sleep(1)  
    geocode_result = gmaps.geocode(i)
    idx = DF['Prop_Addr'].isin([i])
    DF.loc[idx,"latitude"]= geocode_result[0]['geometry']['location']['lat']
    DF.loc[idx,"longitude"] = geocode_result[0]['geometry']['location']['lng']

# Distances from Iowa St U and Airport

import geopy.distance
from geopy.distance import geodesic
IowaStateUcoords = '42.026541, -93.647578'
airportcoords = '41.998823, -93.622318' 
df['IAstateDist'] = ''
df['AirportDist'] = ''
for i in range(len(df.latitude)):
    ix = DF.index.isin([i])
    tup = df.loc[i,'latitude'],df.loc[i,'longitude']
    df.loc[ix,'IAstateDist']=(geodesic(tup, airportcoords).miles)
    df.loc[ix,'AirportDist']= (geodesic(tup, IowaStateUcoords).miles)


totaldf.drop_duplicates(subset=['PID'],inplace = True)

#df.to_csv('finaldf.csv')