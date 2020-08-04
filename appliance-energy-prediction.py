#importing liberies 

import numpy as np
import pandas as pd
import datetime
from  sklearn.linear_model import LinearRegression
import seaborn as sns

#get the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')

df

#rename columns
column_names = {'T1':'Temperature in kitchen area', 'RH_1': 'Humidity in kitchen area', 
                'T2':  'Temperature in living room area', 'RH_2': 'Humidity in living room area',
                'T3': 'Temperature in laundry room area', 'RH_3': ' Humidity in laundry room area', 
                'T4': 'Temperature in office room', 'RH_4': 'Humidity in office room', 
                'T5': 'Temperature in bathroom', 'RH_5': 'Humidity in bathroom', 
                'T6': 'Temperature outside the building', 'RH_6': 'Humidity outside the building', 
                'T7': 'Temperature in ironing room', 'RH_7': 'Humidity outside the building',
               'T8': 'Temperature in teenager room 2', 'RH_8': 'Humidity in teenager room 2',
               'T9': 'Temperature in parents room', 'RH_9': 'Humidity in parents room',
               'T_out': 'Temperature outside (from Chievres weather station)', 'Press_mm_hg': 'Pressure (from Chievres weather station)', 'RH_out': 'Humidity outside (from Chievres weather station)',
               'rv1': ' Random variable 1', 'rv2': ' Random variable 1' }
df = df.rename(columns=column_names)

df

#select a sample of the dataset
simple_linear_reg_df = df[['Appliances', 'Temperature in living room area']].sample(15, random_state=2)

#regression plot
sns.regplot(x = 'Appliances', y= 'Temperature in living room area', data=simple_linear_reg_df)


# convert datetime to the seconds from 1970-01-01 00:00:
df['date'] = pd.to_datetime(df['date'])

#Firstly, we normalise our dataset to a common scale using the min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(columns=["date", "Appliances"])
heating_target = normalised_df['Appliances']

#Now, we split our dataset into the training and testing dataset. 
#Recall that we had earlier segmented the features and target variables.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_df, heating_target, test_size=0.3, random_state=1)

linear_model = LinearRegression()
#fit the model to the training dataset
linear_model.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model.predict(x_test)


#MAE #Mean Absolute Error.py
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 3) 

