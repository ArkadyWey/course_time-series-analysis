import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(filepath_or_buffer="./airline_passengers.csv", index_col="Month", parse_dates=True)

print(df.head())

# Check for nans 
print(df.isna().sum())


# Plot dataset
df.plot()
#plt.show()

# Choose weight parameter
alpha = 0.2 

# Carry out exponentially weighted moving average

df["EWMA"] = df[["Passengers"]].ewm(alpha=alpha, adjust=False).mean() # adjust=False gives classic ewma

# Replot with smoothing
print(df.head())
df.plot()
#plt.show()

# Implement EWMA manually 
# x_bar_{t} = alpha*x_{t} + (1-alpha)*x_bar_{t-1}

def get_manual_ewma(data,alpha):
    manual_ewma = []
    for i,x in enumerate(data):
        if len(manual_ewma) == 0:
            new = x
        else: 
            new = alpha*x + (1-alpha)*manual_ewma[i-1]
        manual_ewma.append(new)
    return manual_ewma


df["manual"] = get_manual_ewma(data=df["Passengers"].to_numpy(), alpha=alpha)

# Replot with manual smoothing
print(df.head())
df.plot()
plt.show()
