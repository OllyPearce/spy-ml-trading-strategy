import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
PRICE_INP = Path("data/cleaned/SPY_cleaned.parquet")
ML_INP = Path("backtest/ml_results_regime_on.parquet")  # produced by regression walk-forward

OUT = Path("backtest/rank_ml_ma200_results.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parameters (conservative)
# -----------------------------
INITIAL_CAPITAL = 10_000.0
MA_WINDOW = 200
COST = 0.0002

# Rank-to-weight mapping (conservative)
# signal_rank is in ~[0,1]; neutral is 0.5
# weight = clip(0.5 + K*(rank-0.5), 0, 1)
K = 1  # smaller = more conservative; try 0.3/0.5/0.8 later

# Volatility targeting
VOL_LOOKBACK = 20
TARGET_VOL_ANNUAL = 0.06  # 6% annual vol target
MAX_WEIGHT = 1.5    # conservative: no leverage, cap at 100% notional

# -----------------------------
# Load data
# -----------------------------
price_df = pd.read_parquet(PRICE_INP).sort_index()
ml_df = pd.read_parquet(ML_INP).sort_index()

df = price_df.join(ml_df, how="left")

adj = df["Adj Close"]
open_px = df["Open"]

# Model output: predicted rank of forward returns (0..1)
signal_rank = df["pred_ret_fwd"].fillna(0.5)  # neutral when missing

# -----------------------------
# Regime (MA200)
# -----------------------------
ma = adj.rolling(MA_WINDOW).mean()
regime_eod = (adj > ma).astype(int)

# Trade at today's open using yesterday's information
regime = regime_eod.shift(1).fillna(0).astype(int)
signal_rank = signal_rank.shift(1).fillna(0.5)

# -----------------------------
# Base weight from rank (conservative)
# -----------------------------
base_weight = 0.5 + K * (signal_rank - 0.5)
base_weight = base_weight.clip(lower=0.0, upper=MAX_WEIGHT)

# Only take exposure in risk-on regime
base_weight = base_weight * regime

# -----------------------------
# Vol targeting (scale base_weight down when vol high)
# Conservative: we only DE-lever (never scale above base_weight)
# -----------------------------
open_ret = open_px.pct_change()
realized_vol_daily = open_ret.rolling(VOL_LOOKBACK).std()
target_vol_daily = TARGET_VOL_ANNUAL / np.sqrt(252)

vol_scale = (target_vol_daily / realized_vol_daily).replace([np.inf, -np.inf], np.nan).fillna(0.0)
vol_scale = vol_scale.clip(lower=0.0, upper=2.0)  # de-lever only

weight = (base_weight * vol_scale).clip(lower=0.0, upper=MAX_WEIGHT)

in_pos = (weight > 0).astype(int)

# -----------------------------
# Backtest (cash + shares, continuous rebalancing)
# -----------------------------
cash = INITIAL_CAPITAL
shares = 0.0
equity = np.zeros(len(df), dtype=float)

for i in range(len(df)):
    px = float(open_px.iloc[i])

    total_equity = cash + shares * px
    target_notional = total_equity * float(weight.iloc[i])
    current_notional = shares * px
    delta = target_notional - current_notional

    if abs(delta) / max(total_equity, 1e-9) > 1e-4:
        # Buy
        if delta > 0:
            spend = min(delta, cash)
            cost = spend * COST
            spend_after_cost = max(spend - cost, 0.0)

            buy_shares = spend_after_cost / px if px > 0 else 0.0
            shares += buy_shares
            cash -= spend  # includes cost

        # Sell
        else:
            sell_notional = min(-delta, current_notional)
            sell_shares = sell_notional / px if px > 0 else 0.0

            proceeds = sell_shares * px
            cost = proceeds * COST
            proceeds_after_cost = max(proceeds - cost, 0.0)

            shares -= sell_shares
            cash += proceeds_after_cost

    equity[i] = cash + shares * px

res = pd.DataFrame(
    {
        "equity": equity,
        "signal_rank": signal_rank,
        "base_weight": base_weight,
        "vol_scale": vol_scale,
        "weight": weight,
        "in_pos": in_pos,
        "regime": regime,
    },
    index=df.index,
).dropna()

# -----------------------------
# Metrics
# -----------------------------
rets = res["equity"].pct_change().fillna(0)
years = (res.index[-1] - res.index[0]).days / 365.25

total = res["equity"].iloc[-1] / res["equity"].iloc[0] - 1
cagr = (res["equity"].iloc[-1] / res["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

vol = rets.std() * np.sqrt(252)
sharpe = (rets.mean() * 252 / vol) if vol > 0 else np.nan

dd = res["equity"] / res["equity"].cummax() - 1
max_dd = dd.min()

turnover = res["weight"].diff().abs().sum()
trades = int(res["in_pos"].diff().abs().fillna(0).sum())

print("\n=== Rank-ML + MA200 (Conservative) ===")
print(f"K={K:.2f}, target_vol={TARGET_VOL_ANNUAL*100:.2f}%, vol_lookback={VOL_LOOKBACK}d (de-lever only)")
print(f"Avg weight: {res['weight'].mean():.3f}  Max weight: {res['weight'].max():.3f}")
print(f"Final: {res['equity'].iloc[-1]:.2f}")
print(f"Total return: {total*100:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Vol: {vol*100:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max DD: {max_dd*100:.2f}%")
print(f"Turnover (sum |Δweight|): {turnover:.2f}")
print(f"Total trades (entries+exits): {trades}")

# -----------------------------
# Save + Plot
# -----------------------------
res.to_parquet(OUT)
print(f"\nSaved results to: {OUT}")

plt.figure()
res["equity"].plot()
plt.title("Equity Curve: Rank-ML + MA200 (Conservative)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.show()
