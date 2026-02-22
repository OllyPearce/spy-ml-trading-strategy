import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
PRICE_INP = Path("data/cleaned/SPY_cleaned.parquet")
ML_INP = Path("backtest/ml_results_regime_on.parquet")

OUT = Path("backtest/ml_ma200_results.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
INITIAL_CAPITAL = 10_000.0
MA_WINDOW = 200
COST = 0.0002

# Percentile-based sizing (robust to uncalibrated probabilities)
PCT_LOW = 0.40  # below this percentile -> 0 exposure
PCT_HIGH = 0.90 # at/above this percentile -> full exposure

# Volatility targeting
VOL_LOOKBACK = 20          # days
TARGET_VOL_ANNUAL = 0.06   # 6% annual vol target
MAX_LEVERAGE = 1.5         # cap scaling
MIN_LEVERAGE = 0.0         # no shorting

# -----------------------------
# Load data
# -----------------------------
price_df = pd.read_parquet(PRICE_INP).sort_index()
ml_df = pd.read_parquet(ML_INP).sort_index()

# Align on dates
df = price_df.join(ml_df, how="left")

adj = df["Adj Close"]
open_px = df["Open"]

# ML probability (already inverted in training, so "higher = more bullish")
prob = df["prob_up_20d"].fillna(0.0)

# -----------------------------
# Regime (MA200)
# -----------------------------
ma = adj.rolling(MA_WINDOW).mean()
regime_eod = (adj > ma).astype(int)

# Use yesterday’s info for trading at today's open
regime = regime_eod.shift(1).fillna(0).astype(int)
prob = prob.shift(1).fillna(0.0)

# -----------------------------
# Prob diagnostics
# -----------------------------
print("\nProb summary (after shifting):")
print(prob.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

# -----------------------------
# Position sizing (percentile-based)
# -----------------------------
p_low = float(prob.quantile(PCT_LOW))
p_high = float(prob.quantile(PCT_HIGH))
den = max(p_high - p_low, 1e-9)

raw_w = (prob - p_low) / den
base_weight = raw_w.clip(lower=0.0, upper=1.0)

# Only take exposure in risk-on regime
base_weight = base_weight * regime

# -----------------------------
# Volatility targeting (on top of base_weight)
# -----------------------------
open_ret = open_px.pct_change()
realized_vol_daily = open_ret.rolling(VOL_LOOKBACK).std()

target_vol_daily = TARGET_VOL_ANNUAL / np.sqrt(252)

vol_scale = (target_vol_daily / realized_vol_daily).replace([np.inf, -np.inf], np.nan).fillna(0.0)
vol_scale = vol_scale.clip(lower=MIN_LEVERAGE, upper=MAX_LEVERAGE)

weight = (base_weight * vol_scale).clip(lower=0.0, upper=MAX_LEVERAGE)

# Binary "in position" indicator (any nonzero weight)
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

    # Rebalance if meaningful difference
    if abs(delta) / max(total_equity, 1e-9) > 1e-4:

        # Buy
        if delta > 0:
            spend = min(delta, cash)
            cost = spend * COST
            spend_after_cost = max(spend - cost, 0.0)

            buy_shares = spend_after_cost / px if px > 0 else 0.0
            shares += buy_shares
            cash -= spend  # includes cost since cost is taken out of spend

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
        "prob": prob,
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

# Turnover proxy: sum of absolute daily changes in weight
turnover = res["weight"].diff().abs().sum()

# Trade count: count entries + exits of "in position"
trades = int(res["in_pos"].diff().abs().fillna(0).sum())

print("\n=== ML + MA200 (Percentile-Weighted + Vol Target) ===")
print(f"Percentiles: PCT_LOW={PCT_LOW:.2f} (p_low={p_low:.4f}), PCT_HIGH={PCT_HIGH:.2f} (p_high={p_high:.4f})")
print(f"Vol targeting: lookback={VOL_LOOKBACK}d, target_vol={TARGET_VOL_ANNUAL*100:.2f}%, max_leverage={MAX_LEVERAGE:.2f}")
print(f"Avg weight: {res['weight'].mean():.3f}")
print(f"Max weight: {res['weight'].max():.3f}")
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
plt.title("ML + MA200 Equity Curve (Percentile-Weighted + Vol Target)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.show()
