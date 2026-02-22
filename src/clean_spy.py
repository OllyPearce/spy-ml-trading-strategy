import pandas as pd
from pathlib import Path

RAW = Path("data/raw/SPY_yfinance.csv")
OUT = Path("data/cleaned/SPY_cleaned.parquet")

df = pd.read_csv(RAW, parse_dates=[0], index_col=0).sort_index()
df.index.name = "Date"

required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Coerce numeric
for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with essential missing values
df = df.dropna(subset=["Adj Close", "Close", "Open", "High", "Low", "Volume"])

# Remove duplicate dates if any
df = df[~df.index.duplicated(keep="last")]

# Keep only required columns in a consistent order
df = df[required].copy()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)

print(f"Saved cleaned data: {OUT} | rows={len(df):,} | from {df.index.min().date()} to {df.index.max().date()}")
print(df.tail())
