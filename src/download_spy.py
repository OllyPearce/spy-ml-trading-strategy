import yfinance as yf
import pandas as pd
from pathlib import Path

OUT = Path("data/raw/SPY_yfinance.csv")

df = yf.download(
    "SPY",
    start="2000-01-01",
    progress=False,
    auto_adjust=False,
)

# If yfinance gives MultiIndex columns (field, ticker), flatten them
if isinstance(df.columns, pd.MultiIndex):
    # columns look like (field, ticker). We only want field.
    df.columns = df.columns.get_level_values(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT)

print(f"Saved {len(df):,} rows to {OUT}")
print(df.tail())
