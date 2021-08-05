import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as web
# import pandas_datareader as data

# from pandas_datareader import data
# from pandas_datareader  import data
import pandas_datareader.data as web
import datetime as dt




# loading data
company = 'TSLA'
print(np)
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

# prepare date
scaler = MinMaxScaler(feature_range = (0,1))