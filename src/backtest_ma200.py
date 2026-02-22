import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INP = Path("data/cleaned/SPY_cleaned.parquet")
OUT = Path("backtest/ma200_results.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
INITIAL_CAPITAL = 10_000.0
MA_WINDOW = 200

# Cost model: fraction of traded notional (spread+slippage estimate)
# 2 bps = 0.0002 is a reasonable toy starting point for SPY
COST_PER_TRADE = 0.0002

TRADING_DAYS = 252

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(INP).sort_index()

# Use Adj Close for signal (dividend/split adjusted),
# and Open for execution price (more realistic next-day execution).
price_signal = df["Adj Close"]
price_exec = df["Open"]

# Optional reference series (not used for PnL directly)
asset_ret = price_exec.pct_change().fillna(0)

# -----------------------------
# Signal (computed at end of day t)
# -----------------------------
ma = price_signal.rolling(MA_WINDOW).mean()

# "Risk-on" if price above MA
signal = (price_signal > ma).astype(int)  # 1 = long, 0 = cash

# Position held on day t is signal from day t-1 (no lookahead)
position = signal.shift(1).fillna(0).astype(int)

# -----------------------------
# Trades
# -----------------------------
trade = position.diff().abs().fillna(0)  # 0->1 or 1->0

# -----------------------------
# Portfolio simulation (cash + shares)
# -----------------------------
cash = INITIAL_CAPITAL
shares = 0.0
equity = np.zeros(len(df), dtype=float)

for i in range(len(df)):
    price = price_exec.iloc[i]

    # Execute trades at today's open
    if trade.iloc[i] > 0:
        # Enter position (buy)
        if position.iloc[i] == 1 and shares == 0.0:
            notional = cash
            cost = notional * COST_PER_TRADE
            invest = notional - cost
            shares = invest / price
            cash = 0.0

        # Exit position (sell)
        elif position.iloc[i] == 0 and shares > 0.0:
            notional = shares * price
            cost = notional * COST_PER_TRADE
            cash = notional - cost
            shares = 0.0

    # Mark-to-market equity
    equity[i] = cash + shares * price

# -----------------------------
# Results dataframe
# -----------------------------
res = pd.DataFrame(
    {
        "AdjClose": price_signal,
        "Open": price_exec,
        "MA200": ma,
        "signal_eod": signal,
        "position": position,
        "trade": trade,
        "asset_ret": asset_ret,  # optional reference
        "equity": equity,
    },
    index=df.index,
).dropna(subset=["MA200"])  # drop warm-up region where MA is undefined

# -----------------------------
# Metrics (MA200)
# -----------------------------
res["port_ret"] = res["equity"].pct_change().fillna(0)

total_return = res["equity"].iloc[-1] / res["equity"].iloc[0] - 1.0
years = (res.index[-1] - res.index[0]).days / 365.25
cagr = (res["equity"].iloc[-1] / res["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

vol = res["port_ret"].std() * np.sqrt(TRADING_DAYS)
mean = res["port_ret"].mean() * TRADING_DAYS
sharpe = mean / vol if vol > 0 else np.nan

running_max = res["equity"].cummax()
drawdown = res["equity"] / running_max - 1.0
max_dd = drawdown.min()

num_trades = int(res["trade"].sum())

# -----------------------------
# Buy & Hold benchmark (invest at first open in the res window)
# -----------------------------
bh_shares = INITIAL_CAPITAL / res["Open"].iloc[0]
bh_equity = bh_shares * res["Open"]

bh_ret = bh_equity.pct_change().fillna(0)
bh_total = bh_equity.iloc[-1] / bh_equity.iloc[0] - 1.0
bh_cagr = (bh_equity.iloc[-1] / bh_equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

bh_vol = bh_ret.std() * np.sqrt(TRADING_DAYS)
bh_mean = bh_ret.mean() * TRADING_DAYS
bh_sharpe = bh_mean / bh_vol if bh_vol > 0 else np.nan

bh_running_max = bh_equity.cummax()
bh_drawdown = bh_equity / bh_running_max - 1.0
bh_max_dd = bh_drawdown.min()

# -----------------------------
# Print
# -----------------------------
print("=== MA200 Backtest (SPY) ===")
print(f"Start: {res.index[0].date()}  End: {res.index[-1].date()}  Years: {years:.2f}")
print(f"Initial: {res['equity'].iloc[0]:.2f}  Final: {res['equity'].iloc[-1]:.2f}")
print(f"Total return: {total_return*100:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Ann. vol: {vol*100:.2f}%")
print(f"Sharpe (rf=0): {sharpe:.2f}")
print(f"Max drawdown: {max_dd*100:.2f}%")
print(f"Trades: {num_trades}")

print("\n=== Buy & Hold (SPY) ===")
print(f"Initial: {INITIAL_CAPITAL:.2f}  Final: {bh_equity.iloc[-1]:.2f}")
print(f"Total return: {bh_total*100:.2f}%")
print(f"CAGR: {bh_cagr*100:.2f}%")
print(f"Ann. vol: {bh_vol*100:.2f}%")
print(f"Sharpe (rf=0): {bh_sharpe:.2f}")
print(f"Max drawdown: {bh_max_dd*100:.2f}%")

# -----------------------------
# Save
# -----------------------------
res.to_parquet(OUT)
print(f"\nSaved results to {OUT}")

# -----------------------------
# Plots
# -----------------------------
plt.figure()
res["equity"].plot(label="MA200")
bh_equity.plot(label="Buy & Hold")
plt.title("Equity Curve Comparison (SPY)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.show()

plt.figure()
drawdown.plot(label="MA200")
bh_drawdown.plot(label="Buy & Hold")
plt.title("Drawdown Comparison (SPY)")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.show()
