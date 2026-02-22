import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/SPY_yfinance.csv", parse_dates=[0], index_col=0)
df.index.name = "Date"

price = df["Adj Close"].dropna()
ret = price.pct_change().dropna()

plt.figure()
price.plot()
plt.title("SPY Adjusted Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

plt.figure()
ret.plot()
plt.title("SPY Daily Returns (Adj Close)")
plt.xlabel("Date")
plt.ylabel("Return")
plt.show()
