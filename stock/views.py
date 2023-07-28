from django.shortcuts import render, redirect
from django.http import HttpResponse

###
import yfinance as yf
import json
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# Create your views here.

def stock(request):
    if request.method == "POST":
        query = request.POST['query']
        if query:
            # Get the stock data
            t = query
            ticker = yf.Ticker(t)
            close_values = ticker.history(period="1y")["Close"].tolist()
            dates = ticker.history(period="1y").index.strftime("%Y-%m-%d").tolist()

            # Create a dictionary with the stock data
            data = {"close_value": close_values, "date": dates, "stock": query}
            # print(data)

            # # Convert the dictionary to a JSON string
            # json_data = json.dumps(data)
            # json_data = json.dumps(t)

            # # Write the JSON string to a file
            # with open("data.json", "w") as file:
            #     file.write(json_data)
            return render(request, 'stockdetail.html', data)
    return render(request,'stockdetail.html')
    
# def search(request):
#     if request.method == "POST":

#         # def scraping(request):
#         ticker = request.POST.get('stock').upper() # name
#         STOCK = ticker
#         RANGE = ('2021-01-01','2021-12-31')    
#         price = si.get_live_price(STOCK)
#         hist_data_close = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['close']]
#         hist_data_adj = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['adjclose']]
#         # print(hist_data_close.head())
            
#         hist_data_close.plot(title=STOCK)
#         plt.savefig('stock/static/images/outputs/graph1.png')
#         # hist_data_adj.plot(title=STOCK)
#         # plt.savefig('stock/static/images/outputs/graph2.png')

#         hist_data_adj = hist_data_adj.rename(columns={'adjclose':STOCK})
#         norma = hist_data_adj / hist_data_adj.iloc[0,:]
#         norma["Rolling_Mean"] = norma[STOCK].rolling(window=20).mean()
#         norma.plot(title="Rolling_Mean")
#         plt.savefig('stock/static/images/outputs/graph2.png')

#         std = norma[STOCK].std()
#         norma['upper'] = norma['Rolling_Mean'] + 0.4*std
#         norma['lower'] = norma['Rolling_Mean'] - 0.4*std
#         norma.loc[:,[STOCK,'upper','lower']].plot(title='Bollinger Band')
#         plt.savefig('stock/static/images/outputs/graph3.png')

#         max_52 = int(hist_data_close.max())
#         min_52 = int(hist_data_close.min())

#         # _thread.start_new_thread(testsets, ())

#         context = {'name': ticker,'price': round(price, 3), 'max_52': max_52, "min_52": min_52}

#         return render(request,'search.html', context)
#         return HttpResponse(request.POST.get('query'))

def analysis(request):
    if request.method == 'GET':
        query = request.GET['query']
    else:
        query = request.POST['query']

    if query:
        # Get the stock data
        t = query
    ticker = t
    STOCK = ticker
    RANGE = ('2021-01-01','2021-12-31')    
    price = si.get_live_price(STOCK)
    hist_data_close = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['close']]
    hist_data_adj = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['adjclose']]
    # print(hist_data_close.head())
        
    hist_data_close.plot(title=STOCK)
    plt.savefig('sp/static/images/outputs/analysis/graph1.png')
    # hist_data_adj.plot(title=STOCK)
    # plt.savefig('stock/static/images/outputs/graph2.png')

    hist_data_adj = hist_data_adj.rename(columns={'adjclose':STOCK})
    norma = hist_data_adj / hist_data_adj.iloc[0,:]
    norma["Rolling_Mean"] = norma[STOCK].rolling(window=20).mean()
    norma.plot(title="Rolling_Mean")
    plt.savefig('sp/static/images/outputs/analysis/graph2.png')

    std = norma[STOCK].std()
    norma['upper'] = norma['Rolling_Mean'] + 0.4*std
    norma['lower'] = norma['Rolling_Mean'] - 0.4*std
    norma.loc[:,[STOCK,'upper','lower']].plot(title='Bollinger Band')
    plt.savefig('sp/static/images/outputs/analysis/graph3.png')

    max_52 = int(hist_data_close.max())
    min_52 = int(hist_data_close.min())

    # _thread.start_new_thread(testsets, ())

    context = {'stock': ticker,'price': round(price, 3), 'max_52': max_52, "min_52": min_52}
    return render(request, 'display.html', context)

def calclstm(request):
    ## todo: show progress bar to peeps ##
    return render(request, 'lstm.html')

def lstm(request):
    if request.method == 'GET':
        query = request.GET['query']
    else:
        query = request.POST['query']

    if os.path.exists(f"lstm_models/{query}.pkl"):

        ### Data Collection
        RANGE = ('2015-01-01','2023-03-22')
        STOCK = query

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
        with open(f"lstm_models/{STOCK}.pkl", 'rb') as f:
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
        fig1 = plt.figure()
        plt.plot(scaler.inverse_transform(close),c='red')
        plt.plot(trainPredictPlot,c='yellow')
        plt.plot(testPredictPlot,c='green')
        plt.savefig('sp/static/images/outputs/lstm/graph2.png')

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
        fig2 = plt.figure()
        plt.plot(day_new,scaler.inverse_transform(close[c:]))
        plt.plot(day_pred,scaler.inverse_transform(lst_output))
        plt.savefig('sp/static/images/outputs/lstm/graph1.png')
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
    else:
        ### Hang request and wait till your cpu pops some corn  ###
        ### Data Collection
        RANGE = ('2018-01-01','2022-08-31')
        STOCK = query

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

        print(data)
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
        print(close)
        print(close.shape)

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

        print(X_train.shape)
        print(y_train.shape)

        print(X_test.shape)
        print(ytest.shape)


        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


        ### Create the Stacked LSTM model
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM

        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')


        model.summary()

        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        ##Transformback to original form
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)


        ### Calculate RMSE performance metrics
        import math
        from sklearn.metrics import mean_squared_error
        math.sqrt(mean_squared_error(y_train,train_predict))


        ### Test Data RMSE
        math.sqrt(mean_squared_error(ytest,test_predict))


        ### Plotting 
        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = np.empty_like(close)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(close)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(close)-1, :] = test_predict
        # plot baseline and predictions  
        fig1 = plt.figure()
        plt.plot(scaler.inverse_transform(close),c='red')
        plt.plot(trainPredictPlot,c='yellow')
        plt.plot(testPredictPlot,c='green')
        plt.savefig('sp/static/images/outputs/lstm/graph2.png')


        len(test_data)

        o = len(test_data) - 100
        x_input=test_data[o:].reshape(1,-1)
        x_input.shape


        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        temp_input


        # demonstrate prediction for next 10 days
        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            
            if(len(temp_input)>100):
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

        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)

        import matplotlib.pyplot as plt

        c = len(close)-time_step
        fig2 = plt.figure()
        plt.plot(day_new,scaler.inverse_transform(close[c:]))
        plt.plot(day_pred,scaler.inverse_transform(lst_output))
        plt.savefig('sp/static/images/outputs/lstm/graph1.png')
              
    context = {'stock': query}

    return render(request, 'lstm.html', context)


## ∨∨∨∨∨∨∨∨∨∨∨∨∨∨ What's this for ∨∨∨∨∨∨∨∨∨∨∨∨∨∨ ##
def analysis1(request):
    if request.method == "GET":
        query = request.GET['query']
        if query:
            # Get the stock data
            t = query
        ticker = t
        STOCK = ticker
        RANGE = ('2021-01-01','2021-12-31')    
        price = si.get_live_price(STOCK)
        hist_data_close = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['close']]
        hist_data_adj = si.get_data(STOCK).loc[RANGE[0]:RANGE[1],['adjclose']]
        # print(hist_data_close.head())
            
        hist_data_close.plot(title=STOCK)
        plt.savefig('sp/static/images/outputs/graph1.png')
        # hist_data_adj.plot(title=STOCK)
        # plt.savefig('stock/static/images/outputs/graph2.png')

        hist_data_adj = hist_data_adj.rename(columns={'adjclose':STOCK})
        norma = hist_data_adj / hist_data_adj.iloc[0,:]
        norma["Rolling_Mean"] = norma[STOCK].rolling(window=20).mean()
        norma.plot(title="Rolling_Mean")
        plt.savefig('sp/static/images/outputs/graph2.png')

        std = norma[STOCK].std()
        norma['upper'] = norma['Rolling_Mean'] + 0.4*std
        norma['lower'] = norma['Rolling_Mean'] - 0.4*std
        norma.loc[:,[STOCK,'upper','lower']].plot(title='Bollinger Band')
        plt.savefig('sp/static/images/outputs/graph3.png')

        max_52 = int(hist_data_close.max())
        min_52 = int(hist_data_close.min())

        # _thread.start_new_thread(testsets, ())

        context = {'name': ticker,'price': round(price, 3), 'max_52': max_52, "min_52": min_52}

        return render(request,'search.html', context)
        return HttpResponse(request.POST.get('query'))
