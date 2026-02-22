import pandas as pd
import numpy as np
from pathlib import Path

INP = Path("data/cleaned/SPY_cleaned.parquet")
OUT = Path("data/features/SPY_features_basic.parquet")

df = pd.read_parquet(INP).sort_index()
price = df["Adj Close"]

df["ret_1d"] = price.pct_change()
df["logret_1d"] = np.log(price).diff()
df["vol_20d"] = df["logret_1d"].rolling(20).std()

running_max = price.cummax()
df["drawdown"] = (price / running_max) - 1.0

df = df.dropna()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)

print(f"Saved features: {OUT} | rows={len(df):,}")
print(df[["ret_1d", "logret_1d", "vol_20d", "drawdown"]].tail())
