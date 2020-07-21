#import libraries 
import numpy as np
import pandas as pd

#loading the fuel dataset from url
url = 'https://raw.githubusercontent.com/WalePhenomenon/climate_change/master/fuel_ferc1.csv'
fuel_data = pd.read_csv(url, error_bad_lines=False)

#Checking the dataset
fuel_data.head()

#Data cleaning #check for duplicate rows
fuel_data.duplicated().any()

#Check for missing values 
df = fuel_data.isnull()
df

#fill the missing values with NaN(Not a Number)
df1 = fuel_data.fillna('NaN')
df1

#count the number of nan(missing) values in each column
#The 'fuel_unit' is the only feature with missing value of 180 

df2 = fuel_data.isnull().sum()
df2

#checking the data summary for 75th percentile of the measure of energy per unit (Fuel_mmbtu_per_unit) 
fuel_data.describe(include='all')

#checking the kurtosis
pd.DataFrame(fuel_data).kurtosis()

#check the value counts of 'fuel_type_code_pudl' to determine the lowest average fuel cost per unit burned
df3 = fuel_data['fuel_type_code_pudl'].value_counts()
df3

#sorting fuel data set to determine the year with the highest average fuel cost per unit delivered
fuel_data.sort_values('fuel_cost_per_unit_delivered', axis = 0, ascending = True, inplace = True, na_position ='last')

fuel_data

