# src/ml_label_engineering.py
import numpy as np
import pandas as pd
from pathlib import Path

INP = Path("data/cleaned/SPY_cleaned.parquet")
OUT = Path("data/features/SPY_ml_dataset.parquet")

HORIZON = 60  # forward days (keep as 60 for the stronger signal)

df = pd.read_parquet(INP).sort_index()

price = df["Adj Close"]

# -----------------------------
# Target (REGRESSION): forward return over HORIZON days
# -----------------------------
fwd_return = price.shift(-HORIZON) / price - 1.0

# Rolling rank of forward returns (0..1)
df["target"] = (
    fwd_return
    .rolling(252)
    .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
)


# -----------------------------
# Features (all info available at time t)
# -----------------------------
df["ret_1d"] = price.pct_change()
df["ret_5d"] = price.pct_change(5)
df["ret_20d"] = price.pct_change(20)
df["ret_60d"] = price.pct_change(60)

df["vol_20d"] = price.pct_change().rolling(20).std()
df["vol_60d"] = price.pct_change().rolling(60).std()

# Volatility regime features
df["vol_ratio"] = df["vol_20d"] / df["vol_60d"]
df["vol_z"] = (
    df["vol_20d"] - df["vol_20d"].rolling(252).mean()
) / df["vol_20d"].rolling(252).std()


ma200 = price.rolling(200).mean()
df["dist_ma200"] = price / ma200 - 1.0

running_max = price.cummax()
df["drawdown"] = price / running_max - 1.0

df["ma_slope"] = ma200.pct_change(20)

# Regime (for filtering in training)
df["regime_ma200"] = (price > ma200).astype(int)

# Medium-term momentum
df["mom_3m"] = price.pct_change(63)   # ~3 months
df["mom_6m"] = price.pct_change(126)  # ~6 months

# Trend strength
ma50 = price.rolling(50).mean()
df["ma50_200"] = ma50 / ma200 - 1.0

df["trend_6m"] = price / price.shift(126) - 1.0
df["trend_12m"] = price / price.shift(252) - 1.0


# -----------------------------
# Drop NaNs (features + target)
# -----------------------------
df = df.dropna()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)

print("Saved ML dataset (REGRESSION):", OUT)
print("Rows:", len(df))
print(df[["target", "regime_ma200", "ret_20d", "vol_20d", "dist_ma200"]].tail())
