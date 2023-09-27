import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv("./sp500_close.csv", index_col=0, parse_dates=True)

print(df.head())

# Get google price
goog = df[["GOOG"]]
goog = goog.dropna("index")
print(goog.head(5))
goog.plot()
#plt.show()

# Get google return
goog_R = goog.pct_change(periods=1)
#goog_R = goog_R.dropna()
print(goog_R.head())

# Get google log return 
goog_ret = np.log(goog_R+1)
goog_ret.plot()
#plt.show()
print(goog_ret.head())

# Take SMA: Simple Moving Average 
goog["SMA-10"] = goog.rolling(window=10).mean() # first result will be 10th entry
print(goog.head(10))
goog.plot()
#plt.show() 

# Take longer SMA
goog["SMA-50"] = goog["GOOG"].rolling(window=50).mean()
print(goog.head(51))
goog.plot()
#plt.show()

# Get data frame of google and apple close prices 
goog_aapl = df[["GOOG","AAPL"]].dropna()
print(goog_aapl.head())

# Get the rolling covariance betweeen the two stocks 
cov = goog_aapl.rolling(window=50).cov()
print(cov.head(102))

# Get the log return 
goog_aapl_ret = np.log(goog_aapl.pct_change(1)+1)

# Get their simple moving average 
goog_aapl_ret["GOOG-SMA-50"] = goog_aapl_ret["GOOG"].rolling(window=50).mean()
goog_aapl_ret["AAPL-SMA-50"] = goog_aapl_ret["AAPL"].rolling(window=50).mean()
goog_aapl_ret.plot()
#plt.show()
print(goog_aapl_ret.head())

# Get the covariance of the two returns 
goog_aapl_cov = goog_aapl_ret[["GOOG","AAPL"]].rolling(window=50).cov()
print(goog_aapl_cov.tail())

# Get the correlation matrix of the two returns 
goog_aapl_corr = goog_aapl_ret[["GOOG","AAPL"]].rolling(window=50).corr()
print(goog_aapl_corr.tail())


