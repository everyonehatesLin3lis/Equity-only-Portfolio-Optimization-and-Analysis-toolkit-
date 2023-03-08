import numpy as np
import pandas_datareader as pdr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
from scipy.stats import norm

start = dt.datetime(2018, 6, 1)#for single pick
now = dt.datetime.now()
akcija=input("Single stock or Multi stocks: ")
def tickers_data(tickers,*args,start,now):#for multi stock
    start = dt.datetime(2022, 10, 1)
    now = dt.datetime.now()
    tickers
    try:
        data = web.DataReader(tickers, 'yahoo', start, now)['Adj Close']
    except ValueError:
        data = pd.DataFrame()
        for y in tickers:
            temp = web.DataReader(y, 'yahoo', start, now)['Adj Close']
            temp = temp.rename(y)
            temp = temp[~temp.index.duplicated()]
            data = data.join(temp, how='outer')
    return data
def single_stock(stock,*args, start, now):
    data = pdr.get_data_yahoo(stock, start, now)
    data.head()
    return data
"""def tickers_data(tickers,*args,start,now):
    data = pdr.get_data_yahoo(tickers, start,now)
    data.head()
    return data"""
def index(stock,start,now):
    stock=['^GSPC']
    index = pdr.get_data_yahoo(stock,start,now)
    index.head()
    return index
def volatility(data):
    high_low = data['High'] - data['Low']
    high_cp = np.abs(data['High'] - data['Close'].shift())
    low_cp = np.abs(data['Low'] - data['Close'].shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(df, axis=1)
    average_true_range = true_range.rolling(14).mean()
    average_true_range = true_range.rolling(14).sum()/14
    fig, ax = plt.subplots()
    average_true_range.plot(ax=ax, color='black')
    ax2 = data['Close'].plot(ax=ax, secondary_y=True, alpha=.3,color='red')
    ax.set_ylabel("ATR-black")
    ax2.set_ylabel("Price-red")
def sharp_ratio(data,portfolio,amountofweights):
    #data.head()
    data
    np.sum(portfolio)
    amountofweights
    np.sum(np.log(data/data.shift())*portfolio, axis=1)
    log_return = np.sum(np.log(data/data.shift())*portfolio, axis=1)
    log_return
    fig, ax = plt.subplots()
    log_return.hist(bins=50, ax=ax)
    log_return.std()
    log_return.mean()
    sharpe_ratio = log_return.mean()/log_return.std()
    sharpe_ratio
    asr = sharpe_ratio*252**.5
    asr
    weight = np.random.random(amountofweights)
    weight /= weight.sum()
    weight
    log_return2 = np.sum(np.log(data/data.shift())*weight, axis=1)
    sharpe_ratio2 = log_return2.mean()/log_return2.std()
    asr2 = sharpe_ratio2*252**.5
    asr2
def Msharp_ratio(data,portfolio):
    data
    log_returns = np.log(data/data.shift())
    log_returns
    amountofweights
    weight = np.random.random(amountofweights)
    weight /= weight.sum()
    weight
    exp_rtn = np.sum(log_returns.mean()*weight)*252
    exp_vol = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov()*252, weight)))
    sharpe_ratio = exp_rtn / exp_vol
    sharpe_ratio
    n = 5000

    weights = np.zeros((n, amountofweights))
    exp_rtns = np.zeros(n)
    exp_vols = np.zeros(n)
    sharpe_ratios = np.zeros(n)

    for i in range(n):
        weight = np.random.random(amountofweights)
        weight /= weight.sum()
        weights[i] = weight

        exp_rtns[i] = np.sum(log_returns.mean()*weight)*252
        exp_vols[i] = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov()*252, weight)))
        sharpe_ratios[i] = exp_rtns[i] / exp_vols[i]
        sharpe_ratios.max()
        sharpe_ratios.argmax()

    fig, ax = plt.subplots()
    ax.scatter(exp_vols, exp_rtns, c=sharpe_ratios)
    ax.scatter(exp_vols[sharpe_ratios.argmax()], exp_rtns[sharpe_ratios.argmax()], c='r')
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    print(weights[sharpe_ratios.argmax()])
def correlation(data):
    data
    returns = data.pct_change()
    returns.tail()
    corr_matrix = returns.corr()
    print(corr_matrix)
    cmap = LinearSegmentedColormap.from_list('', ['red', 'orange', 'yellow', 'chartreuse', 'limegreen'])
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.show()
def linerregression(data,ticker_a,ticker_b):
    data
    log_returns = np.log(data/data.shift())
    ticker_a
    ticker_b
    X = log_returns[ticker_a].iloc[1:].to_numpy().reshape(-1, 1)
    Y = log_returns[ticker_b].iloc[1:].to_numpy().reshape(-1, 1)
    lin_regr = LinearRegression()
    lin_regr.fit(X, Y)
    Y_pred = lin_regr.predict(X)
    alpha = lin_regr.intercept_[0]
    beta = lin_regr.coef_[0, 0]
    fig, ax = plt.subplots()
    ax.set_title("Alpha: " + str(round(alpha, 5)) + ", Beta: " + str(round(beta, 3)))
    ax.scatter(X, Y)
    ax.plot(X, Y_pred, c='r')
    #Tickers data def
def beta(data):
    data
    log_returns=np.log(data/data.shift())
    cov= log_returns.cov()
    var= log_returns['^GSPC'].var()
    print(cov.loc['^GSPC']/var)
def CAPM(data):
    data
    log_returns = np.log(data/data.shift())
    cov = log_returns.cov()
    var = log_returns['^GSPC'].var()
    beta = cov.loc['^GSPC']/var
    #Input here
    risk_free_return=0.02465
    #N here
    market_return=.099
    market_return=risk_free_return+beta*(market_return-risk_free_return)
    print(market_return)
def var(data,weights,initial_investment):
    data
    weights
    initial_investment
    returns = data.pct_change()
    cov_matrix = returns.cov()
    avg_rets=returns.mean()
    port_mean=avg_rets.dot(weights)
    port_stdev=np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    mean_investment = (1+port_mean) * initial_investment
    stdev_investment = initial_investment * port_stdev
    conf_level1 = 0.05
    cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)
    var_1d1 = initial_investment - cutoff1
    var_array = []
    num_days = int(15)
    for x in range(1, num_days+1):
        var_array.append(np.round(var_1d1 * np.sqrt(x),2))
        print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1 * np.sqrt(x),2)))
    plt.xlabel("Day #")
    plt.ylabel("Max portfolio loss (USD)")
    plt.title("Max portfolio loss (VaR) over 15-day period")
    plt.plot(var_array, "r")
    plt.show()
if akcija == 'Single':
    stock = input("Stock: ")
    choise = input("Options with stock: ")
    single_stock(stock,start=start,now=now)
    if choise == 'volatility':
        volatility(single_stock(stock,start=start,now=now))
        plt.show()
elif akcija == 'Multi':
    tickers=[]
    tickers.extend((input("Stocks: ").split(',')))
    portfolio=[]
    portfolio.extend(map(float,input("Weights of stocks: ").split(',')))
    choise = input("Options with stock: ")
    amountofweights=len(portfolio)
    if choise == 'sharp ratio':
        sharp_ratio(tickers_data(tickers,start=start,now=now),portfolio,amountofweights)
        plt.show()
    elif choise == 'Msharp ratio':
        Msharp_ratio(tickers_data(tickers,start=start,now=now),portfolio)
        plt.show()
    elif choise == 'correlation':
        correlation(tickers_data(tickers,start=start,now=now))
        plt.show()
    elif choise == 'linerregression':
        ticker_a = input("1st Stock: ")
        ticker_b = input("2nd Stock: ")
        linerregression(tickers_data(tickers,start=start,now=now),ticker_a,ticker_b)
        plt.show()
    elif choise == 'beta':
        tickers.extend(('^GSPC').split(','))
        beta(tickers_data(tickers,start=start,now=now))
    elif choise == 'CAPM':
        tickers.extend(('^GSPC').split(','))
        CAPM(tickers_data(tickers,start=start,now=now))
    elif choise == 'var':
        initial_investment=float(input("Initial investment: "))
        #weights=np.array([.3,.3,.4])
        weights=np.array(portfolio)
        var(tickers_data(tickers,start=start,now=now),weights,initial_investment)

