import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HistoricalCholesky:

    def __init__(self, df=None, cov_window=252, retraining_interval="y"):
        
        self.stock_df = df
        self.cov_window = cov_window
        self.total_hist_cholesky_df = self.get_historical_cholesky(self.stock_df)
        if retraining_interval == "y":
            self.hist_cholesky_data = self.yearly_data_split(self.total_hist_cholesky_df)
    
    def get_historical_cholesky(self, df=None):
        
        date_count = len(df["date"].unique())
        stock_dim = len(df["ticker"].unique())
        entries_df = pd.DataFrame()
        
        for d in range(1,date_count-self.cov_window+1):
            start_idx = d*stock_dim
            end_idx = (d+self.cov_window)*stock_dim
            temp = df.iloc[start_idx:end_idx]
            temp = temp.pivot(index="date",columns="ticker",values="return")
            cov = temp.cov().values
            cholesky_factor = np.linalg.cholesky(cov).T
            entries = []
            for i in range(stock_dim):
                for j in range(i+1):
                    entries.append(cholesky_factor[j,i])
            entries_array = np.array(entries)
            row = entries_array.reshape(1, -1)
            row_df = pd.DataFrame(row)
            row_df["date"] = pd.to_datetime(temp.index[-1])
            row_df.set_index("date",inplace=True)
            entries_df = pd.concat([entries_df,row_df],axis=0)
        return entries_df

    def yearly_data_split(self, df=None):
        
        data_dict = {}
        year_keys = list(df.index.year.unique())
        for year in year_keys:
            data_dict[year] = df[df.index.year <= year]
        return data_dict

def download_data(tickers,start,end):
    
    data=yf.download(tickers,start=start,end=end)
    data=data.stack().reset_index()
    data.drop(columns=["Close","Open","High","Low","Volume"],inplace=True)
    data.rename(columns={"level_1":"ticker","Date":"date","Adj Close":"close"},inplace=True)
    return data

def preprocess_data(df):
    
    df["return"]=df.groupby("ticker")["close"].pct_change()
    return df

#kelly fractions calculated daily using 2y rolling window

#uses closed form approximation of fractions from Positional Options Trading book
def noncorrelated_kelly(df,window=504):
    
    df["kelly_fraction"] = df.groupby("ticker")["return"].transform(
        lambda x: x.shift(1).rolling(window=window).apply(calculate_kelly_fraction,raw=False))
    df = df.groupby('date').apply(normalize_fractions)
    return df

def calculate_kelly_fraction(x):
    
    mean = x.mean()
    std = x.std()
    if std > 0:
        skew = ((x - mean)**3).mean() / std**3
    else:
        skew = pd.NA
    fraction = (mean / (mean**2 + std**2)) + ((skew*(mean**2)) / (mean**2 + std**2)**3)
    return fraction

def normalize_fractions(group):
    
    df["kelly_fraction"] = df["kelly_fraction"].fillna(0)
    group["kelly_fraction"]=group["kelly_fraction"].clip(lower=0)
    total_fraction = group["kelly_fraction"].sum()
    if total_fraction>1:
        group["kelly_fraction"] = group["kelly_fraction"] / total_fraction
    return group

#applies optimization from Frontiers paper https://www.frontiersin.org/articles/10.3389/fams.2020.577050/full
def correlated_kelly(df,window=504,use_total_balance=True):
    
    date_count = len(df["date"].unique())
    stock_dim = len(df["ticker"].unique())
    df['kelly_fraction']=0
    if use_total_balance:
        #sum of all fractions == 1
        constraints = [{'type': 'eq', 'fun': lambda F: np.sum(F)-1}]
    else:
        #sum of all fractions <=1
        constraints = [{'type': 'ineq', 'fun': lambda F: 1-np.sum(F)}]
    bounds = [(0,1) for _ in range(stock_dim)]
    F_initial = np.array([1/stock_dim]*stock_dim)

    for d in range(1,date_count-window):
        
        start_idx = d*stock_dim
        end_idx = (d+window)*stock_dim
        temp = df.iloc[start_idx:end_idx]
        temp = temp.pivot(index="date",columns="ticker",values="return")
        cov = temp.cov().values
        mean = temp.mean().values
        inv_cov = np.linalg.inv(cov)
        rf_rate = .01/252
        r = np.array([rf_rate]*stock_dim)
    
        def objective(F):
            return -(rf_rate + F.T@(mean-r)-((1/2)*F.T@inv_cov@F))
    
        result = minimize(objective,F_initial,method="SLSQP",
                        bounds=bounds, constraints=constraints)
    
        if result["success"] == True:
            F=result["x"]
        else:
            print(f"Optimization could not be solved between indices {left_index},{right_index}")
            break
    
        #.loc is inclusive of right index, therefore subtract 1
        df.loc[end_idx:end_idx+stock_dim-1, 'kelly_fraction'] = F

        if end_idx%(stock_dim*252)==0:
            print("calculated up to: ",df["date"].iloc[end_idx])
    return df

def daily_return(group):
    
    a=(group["return"] * group["kelly_fraction"]).sum()
    return a

def backtest_kelly(df, tick_dict, initial=1000000):
    
    port_returns = df.groupby("date").apply(daily_return).reset_index(name="port_return")
    trade_df = pd.DataFrame()
    trade_df["date"] = port_returns["date"]
    trade_df["port_return"] = port_returns["port_return"]
    trade_df["port_value"] = initial * ((1+trade_df["port_return"]).cumprod())
    
    return trade_df

def portfolio_performance(df,return_col,value_col,rfr="^TNX",interval=252):
    
    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]

    years = (start_date-end_date).days / interval
    
    mean_return = df[return_col].mean()
    std_return = df[return_col].std()
    start_value = df[value_col].iloc[0]
    end_value = df[value_col].iloc[-1]
    total_return = (end_value - start_value)/end_value
    annualized_return = ((1+total_return)**(1/years)) - 1
    sharpe, annualized_sharpe = calculate_sharpe(df,return_col,rfr,interval) 
    max_drawdown = df[return_col].min()

    print(f"mean return: {mean_return}")
    print(f"std return: {std_return}")
    print(f"max drawdown: {max_drawdown}")
    print(f"total return: {total_return}")
    print(f"annualized return: {annualized_return}")
    print(f"sharpe: {sharpe}")
    print(f"annualized sharpe: {annualized_sharpe}")


def calculate_sharpe(df,return_col,rfr='^TNX',interval=252):
    
    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    
    data = yf.download(rfr, start=start_date, end=end_date)
    data=data.reset_index()
    rate_df = pd.DataFrame()
    rate_df[["date","rate"]] = data[["Date","Adj Close"]]
    port_df = pd.DataFrame()
    port_df[["date","return"]] = df[["date",return_col]] 
    temp_df = pd.merge(port_df,rate_df,on="date",how="outer")
    temp_df["rate"]=temp_df["rate"].fillna(method="ffill")
    temp_df["rate"] = (temp_df["rate"]/interval)/100

    sharpe = ((temp_df["return"]-temp_df["rate"]).mean()) / temp_df["return"].std()
    annualized_sharpe = sharpe * np.sqrt(interval)

    return sharpe,annualized_sharpe

def plot_performance(df):
    plt.plot(df['date'], df['port_value'])
    plt.title('Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value ($millions)')
    plt.show()

    cum_return1 = (df["port_return"]).cumsum()
    cum_return2 = (1+df["port_return"]).cumprod()

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    axs[0].plot(df['date'], cum_return1)
    axs[0].set_title('Portfolio Cumulative Return (CumSum)')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Return')
    
    axs[1].plot(df['date'], cum_return2)
    axs[1].set_title('Portfolio Cumulative Return (CumProd)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Return')
    
    plt.tight_layout()
    plt.show()

def main():
    
    #tickers in DJIA on Oct-31-2004, excludes tickers that became inactive
    tickers = ['WMT', 'INTC', 'MSFT', 'KO', 
            'PG', 'AXP', 'IBM', 'MCD', 
            'JNJ', 'DIS', 'CAT', 'HD', 
            'MRK', 'MMM', 'BA', 'VZ', 
            'JPM', 'HON', 'T', 'HPQ', 
            'RTX', 'PFE', 'GE', 'AIG', 
            'MO', 'C', 'XOM']
    start = '2005-01-01'
    end = '2023-12-31'
    initial_balance = 1000000

    df = download_data(tickers,start,end)
    df1 = preprocess_data(df)

    kelly_1 = noncorrelated_kelly(df1,window=504)
    kelly_2 = correlated_kelly(df1,window=504,use_total_balance=True)
    kelly_3 = correlated_kelly(df1,window=504,use_total_balance=False)

    trade_1 = backtest_kelly(kelly_1,tickers,initial_balance)
    trade_2 = backtest_kelly(kelly_2,tickers,initial_balance)
    trade_3 = backtest_kelly(kelly_3,tickers,initial_balance)

    portfolio_performance(trade_1,"port_return","port_value","^TNX",252)
    portfolio_performance(trade_2,"port_return","port_value","^TNX",252)
    portfolio_performance(trade_3,"port_return","port_value","^TNX",252)
    
    plot_performance(trade_1)
    plot_performance(trade_2)
    plot_performance(trade_3)