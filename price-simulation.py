# Assume that log price is normally distributed.

import numpy as np 
import matplotlib.pyplot as plt

T = 1000 
P0 = 10 
drift = 0.001 # drift 

log_price_last = np.log(P0)
 
log_returns = np.zeros(T)
log_prices  = np.zeros(T)

prices = np.zeros_like(log_prices)

for t in range(T):
    # sample a log return 
    log_return = 0.01*np.random.normal(loc=0.0, scale=1.0, size=None) # normal dist with mean=0 and sd=0.01

    log_price = log_price_last + drift + log_return
    price = np.exp(log_price)

    log_returns[t] = log_return + drift
    log_prices[t]  = log_price
    prices[t] = price

    log_price_last = log_price

plt.figure()
plt.plot(prices)
plt.show() 