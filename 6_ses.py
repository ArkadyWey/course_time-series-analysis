"""
Implement SES - simple exponential smoothing. 
This is essentially same as EWMA - exponentially weighted moving average 
apart from in ewma script we focused on using the algorithm to smooth 
whereas here we focus on using the algorithm to forecast.
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Read data 
df = pd.read_csv("./airline_passengers.csv", index_col="Month", parse_dates=True)

# Begin by smoothing data with exponentially weighted moving average (for comparison)
alpha = 0.2 
df["ewma"] = df[["Passengers"]].ewm(alpha=alpha).mean()
print(df.head())

# Define model class (takes 1D data as input)
ses = SimpleExpSmoothing(df["Passengers"])

# No frequency info detected so have to give it manually - ours is monthly 
print(df.index)
df.index.freq = "MS"
print(df.index)

# Redefine model
ses = SimpleExpSmoothing(endog=df["Passengers"], 
                         initialization_method="legacy-heuristic") # method is old method


# Fit the model (takes hyperparameters)
optimise = False
result = ses.fit(smoothing_level=alpha, optimized=optimise)

# Use model to prefict entire data set
df["ses"] = result.predict(start=df.index[0],end=df.index[-1])
print(df.head())
# Note we have just predictede the training set - 
# Hence prediction should be identical to 'fittedvalues'
print(np.allclose(df["ses"],result.fittedvalues))

# Plot the results
df[["Passengers","ewma","ses"]].plot()
# plt.show() # shifted but that's because we're predicting not smoothing - with holt winters we'll see this is right


# Now split data into train and test and forecast

# Split data (test set is last 12 data points)
n_test = 12
train = df.iloc[:-n_test]
test  = df.iloc[-n_test:]

ses = SimpleExpSmoothing(endog=train["Passengers"], 
                         initialization_method="legacy-heuristic")

result = ses.fit() # with no parameters, this chooses alpha to minimise loss between model and train data

indx_train = df.index <= train.index[-1] # train index is True if df.index is in train index range 
indx_test  = df.index > train.index[-1] # train index is True if df.index is in train index range 

# Use fittedvalues to add the fit to train area of set 
df.loc[indx_train,"ses-fitted"] = result.fittedvalues

# Use predict n times ahead to fit to test area of set
df.loc[indx_test, "ses-fitted"] = result.forecast(n_test)

# Plot 
df[["Passengers", "ses-fitted"]].plot()
plt.show()

# Check hyperparameters - alpha 
print(result.params)
# alpha is near 1 which is copying last known value in the series
# This is just the naive forecast

