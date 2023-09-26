import numpy as np 
import pandas as pd 

import sklearn

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv("./SPY.csv", index_col="Date", parse_dates=["Date"])

print(df.head())

# Generate a naive forecast by using previous target as prediction 

df["Close Prediction"] = df["Close"].shift(1)

print(df.head())

# Create object for target and prediction 
# NB remove first row since NaN

y_true = df["Close"][1:]
y_pred = df["Close Prediction"][1:]

print(y_true)
print(y_pred) # these variable names are consistent with sklearn

# Metrics 

# SSE 
sse = sum((y_true-y_pred)**2)
print("sse: {}".format(sse))

# MSE 
mse = sklearn.metrics.mean_squared_error(y_true=y_true,y_pred=y_pred)
print("mse: {}".format(mse))
mse_myself = sse/len(y_true)
print("mse_myself: {}".format(mse_myself))
print("check: {}".format(mse-mse_myself))

# RMSE
rmse = sklearn.metrics.mean_squared_error(y_true=y_true,y_pred=y_pred,squared=False)
print("rmse: {}".format(rmse))
rmse_myself = mse_myself**(1/2)
print("rmse_myself: {}".format(rmse_myself))
print("check: {}".format(rmse-rmse_myself))

# MAE 
mae = sklearn.metrics.mean_absolute_error(y_true=y_true,y_pred=y_pred)
print("mae: {}".format(mae))
mae_myself = sum(abs(y_true-y_pred))/len(y_true)
print("mae_myself: {}".format(mae_myself))
print("check: {}".format(mae-mae_myself))

# r2e
r2s = sklearn.metrics.r2_score(y_true=y_true,y_pred=y_pred)
print("r2s: {}".format(r2s))
r2s_myself = 1 - sse/sum((y_true-y_true.mean())**2)
print("r2s_myself: {}".format(r2s_myself))
print("check: {}".format(r2s-r2s_myself))

# mape 
mape = sklearn.metrics.mean_absolute_percentage_error(y_true=y_true,y_pred=y_pred)
print("mape: {}".format(mape))
mape_myself = sum(abs(y_true-y_pred)/y_true)/len(y_true)
print("mape_myself: {}".format(mape_myself))
print("check: {}".format(mape-mape_myself))

# sMAPE
num = abs(y_true-y_pred)
den = (abs(y_true)+abs(y_pred))/2
smape = sum(num/den)/len(y_true)
print("smape: {}".format(smape))
