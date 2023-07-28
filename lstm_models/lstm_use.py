# -*- coding: utf-8 -*-
"""Model1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WI_iNMdjyYFM6sQysECSnK0Hm4Ngp-G7
"""

# pip install yahoo_fin

"""### Stock Market Prediction And Forecasting Using Stacked LSTM"""

# STACKED LSTM MODEL

### Data Collection
RANGE = ('2015-01-01','2022-12-31')
STOCK = 'TATASTEEL.NS'

import yahoo_fin.stock_info as si
import pandas as pd


# Creating Empty DF
dates = pd.date_range(RANGE[0],RANGE[1])
emptyDF = pd.DataFrame(index=dates)


# Historical Data
hist_data = si.get_data(STOCK,start_date=RANGE[0],end_date=RANGE[1])

# removing ticker col
hist_data = hist_data.iloc[:,:-1]

data = emptyDF.join(hist_data)

# Droping Na
data = data.dropna()

# print(data)
# # Income Statement
# i_data = si.get_income_statement(TICKER)
# Transforming and Sorting Data wise
# i_data = i_data.transpose()[::-1]
# data = data.join(i_data)

# data.iloc[:,6:]=data.iloc[:,6:].ffill()
# data.iloc[:,6:]=data.iloc[:,6:].bfill()


# data.dropna(how='all', axis=1, inplace=True)
# data = data.dropna(how='any')
# data.to_csv('data.csv')

# Extracting Close
close = data.reset_index()['close']
# print(close)
# print(close.shape)

# Plotting The Graph
import matplotlib.pyplot as plt
# close.plot(title="Stock Price")
# plt.xlabel("nth day")
# plt.ylabel("INR")

# Scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

close = scaler.fit_transform(close.values.reshape(-1,1))

print(close)

##splitting dataset into train and test split
training_size=int(len(close)*0.70)

test_size=len(close)-training_size

print(training_size,test_size)

train_data,test_data=close[0:training_size,:],close[training_size:len(close),:1]

# convert an array of values into a dataset matrix

import numpy as np
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# pip install tensorflow

### Create the Stacked LSTM model
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM

# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error',optimizer='adam')

# model.summary()

# model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
import pickle

# Load the pickled model
with open(f"{STOCK}.pkl", 'rb') as f:
    model = pickle.load(f)


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
print(train_predict.shape,test_predict.shape)
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(close)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(close)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(close)-1, :] = test_predict
# plot baseline and predictions
# plt.plot(scaler.inverse_transform(close),c='red')
# plt.plot(trainPredictPlot,c='yellow')
# plt.plot(testPredictPlot,c='green')
# plt.savefig('base.png')

len(test_data)

o = len(test_data) - time_step
x_input=test_data[o:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print(temp_input)

temp_input

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=time_step
i=0
while(i<30):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((-1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((-1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

day_new=np.arange(1,time_step + 1)
day_pred=np.arange(time_step + 1,time_step + 1 + 30)

import matplotlib.pyplot as plt
c = len(close)-time_step
plt.plot(day_new,scaler.inverse_transform(close[c:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

x1 = day_new.tolist()
y1 = scaler.inverse_transform(close[c:])
y1 = [item for sublist in y1 for item in sublist]

x2 = day_pred.tolist()
y2 = scaler.inverse_transform(lst_output)
y2 = [item for sublist in y2 for item in sublist]

y1 = list(map(lambda x: round(x,2),y1))
y2 = list(map(lambda x: round(x,2),y2))

# plt.savefig('new.png')
# df3=close.tolist()
# df3.extend(lst_output)
# plt.plot(df3[c:])
# print(lst_output)