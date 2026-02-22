import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/features/SPY_features_basic.parquet")

plt.figure()
df["logret_1d"].hist(bins=80)
plt.title("SPY Daily Log Returns (Histogram)")
plt.xlabel("log return")
plt.ylabel("count")
plt.show()

plt.figure()
df["vol_20d"].plot()
plt.title("SPY 20D Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("vol")
plt.show()

plt.figure()
df["drawdown"].plot()
plt.title("SPY Drawdown (Adj Close)")
plt.xlabel("Date")
plt.ylabel("drawdown")
plt.show()
