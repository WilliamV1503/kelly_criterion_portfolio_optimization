import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def download_data(tickers,start,end):
    
    data=yf.download(tickers,start=start,end=end)
    data=data.stack().reset_index()
    data.drop(columns=["Close","Open","High","Low","Volume"],inplace=True)
    data.rename(columns={"level_1":"ticker","Date":"date","Adj Close":"close"},inplace=True)
    
    return data

def preprocess_data(df):
    
    df["return"]=df.groupby("ticker")["close"].pct_change()
    df["date"]=pd.to_datetime(df["date"])
    returns_df = df.pivot(index = "date", columns = "ticker", values = "return")
    #returns_df.reset_index(inplace = True)
    
    return returns_df

#uses gaussian_kde object to estimate distribution of returns for each stock and resample
def resample_with_kde(df, n):
   
    kde = gaussian_kde(df.dropna().values.T)
    resampled_data = kde.resample(n).T
    resampled_df = pd.DataFrame(resampled_data, columns = df1.columns)

    return resampled_df


#uses cholesky factor of historial correlation matrix to correlate resampled returns
def correlate_returns(standardized_returns_df, corr_matrix):
    
    returns_array = standardized_returns_df.values
    lt_cholesky = np.linalg.cholesky(corr_matrix)
    correlated_returns_array = (lt_cholesky @ returns_array.T).T
    correlated_returns_df = pd.DataFrame(correlated_returns_array, 
                                         columns = standardized_returns_df.columns)
    correlated_returns_df.index = standardized_returns_df.index
    
    return correlated_returns_df

def simulate_returns(hist_df, n):
    
    corr_matrix = np.array(hist_df.corr())
    resampled_returns = resample_with_kde(hist_df, n)
    stdev = resampled_returns.std()
    mean = resampled_returns.mean()
    standardized_resampled_returns = (resampled_returns - mean) / stdev
    corr_standardized_resampled_returns = correlate_returns(standardized_resampled_returns, 
                                                            corr_matrix)
    corr_resampled_returns = (corr_standardized_resampled_returns * stdev) + mean

    return corr_resampled_returns
    
def main():

    #test
    tickers = ["AAPL", "AMZN", "TSLA", "NVDA"]
    start = '2012-01-01'
    end = '2022-12-31'
    df = download_data(tickers,start,end)
    df1 = preprocess_data(df)

    simulated_returns = simulate_returns(df1, 504)
    print(simulated_returns)
    


