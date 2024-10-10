import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker=input("enter ticker: ")
data=yfinance.download(ticker,start='2024-09-09',end='2024-10-10',interval='15m')
data.dropna()

"""def AddEquity():
    
def RemoveEquity():"""

def ATR(DF,n=14):
    df=DF.copy()
    df["H-L"]=df["High"]-df["Low"]
    df["H-PC"]=df["High"]-df["Adj Close"].shift(1)
    df["L-PC"]=df["Low"]-df["Adj Close"].shift(1)
    df["TR"]=df[["H-L","H-PC","L-PC"]].max(axis=1,skipna=False)
    df["ATR"]=df["TR"].ewm(com=n,min_periods=n).mean()
    
    return df["ATR"]

def MACD(df):
    df['12 Day EMA']=df['Adj Close'].ewm(span=12,min_periods=12).mean()
    df['26 Day EMA']=df['Adj Close'].ewm(span=26,min_periods=26).mean()
    df['MACD']=df['12 Day EMA']-df['26 Day EMA']
    df['Signal Line (MACD)']=df['MACD'].ewm(span=9,min_periods=9).mean()
    
    return df["MACD"]

def RSI(DF,n=14):
    df=DF.copy()
    df["Change"]=df["Adj Close"]-df["Adj Close"].shift(1)
    df["Gain"]=np.where(df["Change"]>=0,df["Change"],0)
    df["Loss"]=np.where(df["Change"]<0,-1*df["Change"],0)
    df["Avg Gain"]=df["Gain"].ewm(alpha=1/n,min_periods=n).mean()
    df["Avg Loss"]=df["Loss"].ewm(alpha=1/n,min_periods=n).mean()
    df["RS"]=df["Avg Gain"]/df["Avg Loss"]
    
    df["RSI"]=100-(100/(1+df["RS"]))
    
    return df["RSI"]

def BolBands(DF,n=20):
    df=DF.copy()
    df["Middle Band"]=df["Adj Close"].rolling(window=n).mean()
    df["Upper Band"]=df["Middle Band"]+2*df["Adj Close"].rolling(window=n).std(ddof=0)
    df["Lower Band"]=df["Middle Band"]-2*df["Adj Close"].rolling(window=n).std(ddof=0)
    df["BB Width"]=df["Upper Band"]-df["Lower Band"]
    
    return df[["Upper Band","Lower Band","Middle Band","BB Width"]]

def ADX(DF, n=20):
    df = DF.copy()
    df["UpMove"] = df["High"] - df["High"].shift(1)
    df["DownMove"] = df["Low"].shift(1) - df["Low"]
    df["ATR"] = ATR(df, n)
    df["+DM"] = np.where((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0)
    df["-DM"] = np.where((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["UpMove"], 0)
    df["+DI"] = 100 * ((df["+DM"] / df["ATR"]).ewm(span=n, min_periods=n).mean())
    df["-DI"] = 100 * ((df["-DM"] / df["ATR"]).ewm(span=n, min_periods=n).mean())
    df["ADX"] = 100 * abs((df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])).ewm(span=n, min_periods=n).mean()
    return df[['+DM','-DM','ADX','+DI','-DI']]

data['ATR'] = ATR(data)
data['MACD'] = MACD(data)
data['RSI'] = RSI(data)
data[["Upper Band","Lower Band","Middle Band","BB Width"]] = BolBands(data)
data[['+DM','-DM','ADX','+DI','-DI']]=ADX(data)

data.dropna(how='any',axis=0,inplace=True)
    
def entrySignal(data):
    condition = (
        (data['ADX'] > 30) &
        (data['+DI']>data['-DI']) &
        (
            (data['MACD'] >= data['Signal Line (MACD)']) |
            (data['RSI'] <= 28) |
            (data['Adj Close'] <= data['Lower Band'])
        )
    )
    return condition

def exitSignal(data):
    condition = (
        (data['ADX'] > 30) &
        (data['+DI'] < data['-DI']) &
        (
            (data['MACD'] <= data['Signal Line (MACD)']) |
            (data['RSI'] >= 72) |
            (data['Adj Close'] >= data['Upper Band'])
        )
    )
    return condition

data['Entry_Signal'] = entrySignal(data)
data['Exit_Signal'] = exitSignal(data)

data['Position'] = 0

for i in range(1, len(data)):
    if data['Entry_Signal'].iloc[i] and data['Position'].iloc[i-1] == 0:
        data['Position'].iloc[i] = 1
    elif data['Exit_Signal'].iloc[i] and data['Position'].iloc[i-1] == 1:
        data['Position'].iloc[i] = 0
    else:
        data['Position'].iloc[i] = data['Position'].iloc[i-1]
        
data['Market_Returns'] = data['Adj Close'].pct_change()
data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

plt.figure(figsize=(14,7))
plt.plot(data['Cumulative_Market_Returns'], label='Market Returns')
plt.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns')
plt.title('Strategy Performance vs. Market Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()