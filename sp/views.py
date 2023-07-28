from django.shortcuts import render
import yfinance as yf
import matplotlib.pyplot as plt

 #  isme saare keys h
# Create your views here.

def home(request):


    sensex = yf.Ticker('^BSESN')
    # print(sensex.fast_info)
    sensex = sensex.fast_info['last_price']
    # print(sensex.fast_info)
    sensex = round(sensex,2)

    nifty= yf.Ticker('^NSEI')
    niftybank= yf.Ticker('^NSEBANK')
    niftyIT = yf.Ticker('^CNXIT')
    infi = yf.Ticker('INFY.NS')
    reliance = yf.Ticker('RELIANCE.NS')
    hdfc = yf.Ticker('HDFCBANK.NS')
    asianpaint = yf.Ticker('ASIANPAINT.NS')
    niftymid = yf.Ticker('NIFTY_MIDCAP_100.NS')
    nifty150 = yf.Ticker('NIFTY_MIDCAP_150.NS')
    # return HttpResponse("Hello, world. I didn't want to wake up. I was having a much better time asleep.")
    nifty=round(nifty.fast_info['last_price'],2)
    niftybank=round(niftybank.fast_info['last_price'],2)
    niftyIT=round(niftyIT.fast_info['last_price'],2)
    infi=round(infi.fast_info['last_price'],2)
    reliance = round(reliance.fast_info['last_price'],2)
    hdfc = round(hdfc.fast_info['last_price'],2)
    asianpaint = round(asianpaint.fast_info['last_price'],2)
    niftymid = round(niftymid.fast_info['last_price'],2)
    nifty150 = round(nifty150.fast_info['last_price'],2)

# all stocks 
    tatapower = yf.Ticker("TATAPOWER.NS")
    mrf = yf.Ticker("MRF.NS")
    drreddy = yf.Ticker("DRREDDY.NS")
    wipro = yf.Ticker("WIPRO.NS")
    tatasteel = yf.Ticker("TATASTEEL.NS")
    tatapower = round(tatapower.fast_info['last_price'],3)
    mrf = round(mrf.fast_info['last_price'],3)
    drreddy = round(drreddy.fast_info['last_price'],3)
    wipro = round(wipro.fast_info['last_price'],3)
    tatasteel = round(tatasteel.fast_info['last_price'],3)

    
    context = {'sensex': sensex,'nifty150':nifty150,'niftymid':niftymid,'nifty': nifty, 'niftybank': niftybank, "niftyIT": niftyIT , "infi":infi ,'reliance':reliance,'hdfc':hdfc,'asianpaint':asianpaint,'tatapower':tatapower,"tatasteel":tatasteel,"drreddy":drreddy,'wipro':wipro,"mrf":mrf }
          
    return render(request,'index1.html',context)

def stocklist(request):
    return render(request,'allstocklist.html')

def stockdetail(request):
    return render(request,'stockdetail.html')

def portfolio(request):
    import yfinance as yf
    import matplotlib.pyplot as plt

    df = yf.Ticker("BAJFINANCE.NS")

    df.fast_info

    df.fast_info['previousClose']

    prices = []
    tickers = ["TATASTEEL.NS", "AXISBANK.NS", "RELIANCE.NS", "SBIN.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "COLPAL.NS", "GODREJCP.NS", "TITAN.NS", "ASIANPAINT.NS" ,"ADANIENT.NS"]
    total = []
    amounts = [2,3,4,5,1,3,2,4,5,1,3]
    for i in tickers:
        df = yf.Ticker(i)
        price_close = df.fast_info['previousClose']
        prices.append(price_close)
        index = tickers.index(i)
        total.append(price_close* amounts[index])

    prices
    index
    total
    fig, ax = plt.subplots()

    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_title(" PORTFOLIO VISUALIZER", color='#EF6C35', fontsize=20)

    plt.pie(total,labels = tickers, autopct='%1.1f%%',pctdistance = 0.6)
    plt.savefig('sp/static/images/outputs/portfolio/graph1.png')

    tatapower = yf.Ticker("TATAPOWER.NS")
    mrf = yf.Ticker("MRF.NS")
    drreddy = yf.Ticker("DRREDDY.NS")
    wipro = yf.Ticker("WIPRO.NS")
    tatasteel = yf.Ticker("TATASTEEL.NS")
    tatapower = round(tatapower.fast_info['last_price'],3)
    mrf = round(mrf.fast_info['last_price'],3)
    drreddy = round(drreddy.fast_info['last_price'],3)
    wipro = round(wipro.fast_info['last_price'],3)
    tatasteel = round(tatasteel.fast_info['last_price'],3)


    ############
    import yfinance as yf
    import pandas as pd
    import numpy as np

    port_list = ["TATASTEEL.NS", "AXISBANK.NS", "RELIANCE.NS", "SBIN.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "COLPAL.NS", "GODREJCP.NS", 
              "TITAN.NS", "ASIANPAINT.NS" ,"ADANIENT.NS"]

    # Define the list of tickers
    tickers = ["TATASTEEL.NS", "AXISBANK.NS", "RELIANCE.NS", "SBIN.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "COLPAL.NS", "GODREJCP.NS", 
                "TITAN.NS", "ASIANPAINT.NS" ,"ADANIENT.NS"]
    risk_free_rate = 0.0125

    # Download stock data for all tickers
    close_prices = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        stock_data = yf.Ticker(ticker).history(start="2010-01-01", end="2023-03-21")
        close_prices[ticker] = stock_data["Close"]

    # Print the DataFrame containing the close prices for all tickers
    mult_df = close_prices
    (mult_df / mult_df.iloc[0] * 100).plot(figsize=(16, 9)).figure.savefig('sp/static/images/outputs/portfolio/graph2.png')
    returns = np.log(mult_df / mult_df.shift(1))
    mean_ret = returns.mean() * 252 # 252 average trading days per year
    returns.cov() * 252
    returns.corr()
    weights = np.random.random(11)
    weights /= np.sum(weights)
    print('Weights :', weights)
    print('Total Weight :', np.sum(weights))
    np.sum(weights * returns.mean()) * 252
    np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    p_ret = [] # Returns list
    p_vol = [] # Volatility list
    p_SR = [] # Sharpe Ratio list
    p_wt = [] # Stock weights list


    for x in range(10000):
        # Generate random weights
        p_weights = np.random.random(11)
        p_weights /= np.sum(p_weights)
        
        # Add return using those weights to list
        ret_1 = np.sum(p_weights * returns.mean()) * 252
        p_ret.append(ret_1)
        
        # Add volatility or standard deviation to list
        vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
        p_vol.append(vol_1)
        
        # Get Sharpe ratio
        SR_1 = (ret_1 - risk_free_rate) / vol_1
        p_SR.append(SR_1)
        
        # Store the weights for each portfolio
        p_wt.append(p_weights)
        
    # Convert to Numpy arrays
    p_ret = np.array(p_ret)
    p_vol = np.array(p_vol)
    p_SR = np.array(p_SR)
    p_wt = np.array(p_wt)

    p_ret, p_vol, p_SR, p_wt

    # Create a dataframe with returns and volatility
    ports = pd.DataFrame({'Return': p_ret, 'Volatility': p_vol})

    ports.plot(x='Volatility', y='Return', kind='scatter', figsize=(16, 9)).figure.savefig('sp/static/images/outputs/portfolio/graph3.png')

    SR_idx = np.argmax(p_SR)

    # Find the ideal portfolio weighting at that index
    i = 0
    while i < 11:
        print("Stock : %s : %2.2f" % (port_list[i], (p_wt[4296][i] * 100)))
        i += 1
        
    # Find volatility of that portfolio
    print("\nVolatility :", p_vol[4296])
        
    # Find return of that portfolio
    print("Return :", p_ret[4296])


    ## side bar 
    context = {'tatapower':tatapower,"tatasteel":tatasteel,"drreddy":drreddy,'wipro':wipro,"mrf":mrf,"pvol":p_vol[4296]*100,"pret":p_ret[4296]*100,"wt":p_wt[4296]}

    return render(request,'portfolio.html',context)