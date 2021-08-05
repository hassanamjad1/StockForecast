import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as web
# import pandas_datareader as data

# from pandas_datareader import data
# from pandas_datareader  import data
import pandas_datareader.data as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# loading data
company = 'TSLA'
print(np)
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

# prepare date
scaler = MinMaxScaler(feature_range = (0,1))



# predicting the closing price
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#predicted price in the next 60 days
prediction_days = 60

x_train = [] #empty lists
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])


#using numpy library
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building The Model of neural Network
model = Sequential()

model.add(LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) # Prediction of the next closur

#compiling thr model
model.compile(optimizer = 'adam', loss = 'mean_squared_err' )

#fitting the model on trainng data
model.fit(x_train, y_train, epochs=25, batch_size=32)