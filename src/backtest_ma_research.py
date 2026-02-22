
"""

MA Regime Strategy Research Script

Implements:
- Risk-free rate (cash yield) on uninvested cash
- Volatility targeting (de-leveraging only; never lever above allocation)
- Multiple MA windows (100, 150, 200, 250)
- Partial allocation (e.g. 50% max exposure)
- Walk-forward testing via contiguous evaluation blocks (e.g., 5-year blocks)
- Parameter stability analysis across window/allocation + block metrics

Assumptions / Conventions:
- Signal computed on Adj Close at end of day t
- Trades/rebalances executed at next day Open (position uses signal.shift(1))
- Mark-to-market using Open prices (Open-to-Open style)
- Trading costs charged on traded notional at execution Open
- Cash earns a constant annualized risk-free rate (toy; replace later with SONIA/T-bills series)

Run:
  python src/backtest_ma_research.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
INP = Path("data/cleaned/SPY_cleaned.parquet")
OUT_DIR = Path("backtest/ma_research")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Global constants / defaults
# -----------------------------
TRADING_DAYS = 252
DEFAULT_COST_PER_TRADE = 0.0002  # 2 bps
DEFAULT_RF_ANNUAL = 0.02         # 2% annual cash yield (toy; replace with series later)
DEFAULT_VOL_TARGET = 0.10        # 10% annual vol target
DEFAULT_VOL_LOOKBACK = 20        # days, based on Open-to-Open returns

MA_WINDOWS = [100, 150, 200, 250]
ALLOCATIONS = [0.5, 1.0]         # partial allocation options


@dataclass(frozen=True)
class Config:
    initial_capital: float = 10_000.0
    cost_per_trade: float = DEFAULT_COST_PER_TRADE
    rf_annual: float = DEFAULT_RF_ANNUAL
    use_vol_target: bool = True
    vol_target_annual: float = DEFAULT_VOL_TARGET
    vol_lookback: int = DEFAULT_VOL_LOOKBACK
    years_per_block: int = 5      # for walk-forward stability blocks


@dataclass(frozen=True)
class StrategyParams:
    ma_window: int
    allocation: float  # max fraction of equity allocated to SPY when risk-on (e.g., 0.5 or 1.0)


def daily_rf_rate(rf_annual: float) -> float:
    return (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return dict(total_return=np.nan, cagr=np.nan, ann_vol=np.nan, sharpe_rf0=np.nan, max_dd=np.nan)

    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    ann_mean = rets.mean() * TRADING_DAYS
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = dd.min()

    return dict(
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe_rf0=sharpe,
        max_dd=max_dd,
    )


def buy_and_hold_open(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    open_px = df["Open"].sort_index()
    shares = initial_capital / open_px.iloc[0]
    return shares * open_px


def buy_and_hold_with_cash_yield(df: pd.DataFrame, initial_capital: float, rf_annual: float) -> pd.Series:
    """
    Fairer benchmark if you assume cash earns rf: invest fully in SPY immediately (no cash after).
    So this is identical to buy-and-hold unless you model margin/cash. Kept here for clarity.
    """
    return buy_and_hold_open(df, initial_capital)


def walk_forward_blocks(index: pd.DatetimeIndex, years_per_block: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = index.min()
    end = index.max()
    blocks = []
    cur = pd.Timestamp(start.year, start.month, start.day)

    while cur < end:
        nxt = cur + pd.DateOffset(years=years_per_block)
        blocks.append((cur, min(nxt, end)))
        cur = nxt

    return blocks


def metrics_by_blocks(equity: pd.Series, blocks: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for (a, b) in blocks:
        seg = equity.loc[(equity.index >= a) & (equity.index < b)]
        if len(seg) < 10:
            continue
        m = compute_metrics(seg)
        rows.append({"block_start": a.date(), "block_end": b.date(), **m})
    return pd.DataFrame(rows)


def backtest_ma(df: pd.DataFrame, cfg: Config, params: StrategyParams) -> pd.DataFrame:
    df = df.sort_index().copy()

    adj = df["Adj Close"]
    open_px = df["Open"]

    # Regime signal
    ma = adj.rolling(params.ma_window).mean()
    signal_eod = (adj > ma).astype(int)
    pos_on_day = signal_eod.shift(1).fillna(0).astype(int)  # trade at open t using info up to t-1 close

    # Vol targeting input
    open_ret = open_px.pct_change()
    daily_vol = open_ret.rolling(cfg.vol_lookback).std()
    vol_target_daily = cfg.vol_target_annual / np.sqrt(TRADING_DAYS)

    # Exposure weight (never > allocation; de-lever only)
    if cfg.use_vol_target:
        vol_scale = (vol_target_daily / daily_vol).clip(upper=1.0)
        weight = (params.allocation * vol_scale).fillna(0.0)
    else:
        weight = pd.Series(params.allocation, index=df.index, dtype=float)

    weight = weight * pos_on_day.astype(float)  # 0 when in cash regime

    rf_daily = daily_rf_rate(cfg.rf_annual)

    cash = cfg.initial_capital
    shares = 0.0

    equity = np.zeros(len(df), dtype=float)
    cash_arr = np.zeros(len(df), dtype=float)
    shares_arr = np.zeros(len(df), dtype=float)
    w_arr = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        px = float(open_px.iloc[i])

        # Cash yield on whatever cash you hold
        cash *= (1.0 + rf_daily)

        current_equity = cash + shares * px
        target_notional = current_equity * float(weight.iloc[i])
        current_notional = shares * px
        delta_notional = target_notional - current_notional

        # Rebalance to target notional
        if abs(delta_notional) / max(current_equity, 1e-9) > 1e-6:
            if delta_notional > 0:
                spend = min(delta_notional, cash)
                cost = spend * cfg.cost_per_trade
                spend_after_cost = max(spend - cost, 0.0)
                buy_shares = spend_after_cost / px if px > 0 else 0.0
                shares += buy_shares
                cash -= spend
            else:
                sell_notional = min(-delta_notional, current_notional)
                sell_shares = sell_notional / px if px > 0 else 0.0
                proceeds = sell_shares * px
                cost = proceeds * cfg.cost_per_trade
                proceeds_after_cost = max(proceeds - cost, 0.0)
                shares -= sell_shares
                cash += proceeds_after_cost

        equity[i] = cash + shares * px
        cash_arr[i] = cash
        shares_arr[i] = shares
        w_arr[i] = float(weight.iloc[i])

    res = pd.DataFrame(
        {
            "Open": open_px,
            "AdjClose": adj,
            "MA": ma,
            "signal_eod": signal_eod,
            "position": pos_on_day,
            "weight": w_arr,
            "equity": equity,
            "cash": cash_arr,
            "shares": shares_arr,
        },
        index=df.index,
    )

    # Warmup removal for MA and vol lookback
    warmup = max(params.ma_window, cfg.vol_lookback)
    res = res.iloc[warmup:].copy()

    return res


def dd(series: pd.Series) -> pd.Series:
    rm = series.cummax()
    return series / rm - 1.0


def main():
    cfg = Config(
        initial_capital=10_000.0,
        cost_per_trade=DEFAULT_COST_PER_TRADE,
        rf_annual=DEFAULT_RF_ANNUAL,
        use_vol_target=True,
        vol_target_annual=DEFAULT_VOL_TARGET,
        vol_lookback=DEFAULT_VOL_LOOKBACK,
        years_per_block=5,
    )

    df = pd.read_parquet(INP).sort_index()

    blocks = walk_forward_blocks(df.index, years_per_block=cfg.years_per_block)

    summary_rows = []
    block_rows = []

    bh_equity_full = buy_and_hold_open(df, cfg.initial_capital)

    for w in MA_WINDOWS:
        for alloc in ALLOCATIONS:
            params = StrategyParams(ma_window=w, allocation=alloc)

            res = backtest_ma(df, cfg, params)
            eq = res["equity"]

            # Match buy&hold to same window
            bh_equity = bh_equity_full.loc[eq.index[0]: eq.index[-1]]

            m = compute_metrics(eq)
            bh_m = compute_metrics(bh_equity)

            run_tag = f"ma{w}_alloc{int(alloc*100)}_vt{int(cfg.use_vol_target)}"
            res_path = OUT_DIR / f"res_{run_tag}.parquet"
            res.to_parquet(res_path)

            summary_rows.append(
                {
                    "run_tag": run_tag,
                    "ma_window": w,
                    "allocation": alloc,
                    "use_vol_target": cfg.use_vol_target,
                    "vol_target": cfg.vol_target_annual if cfg.use_vol_target else np.nan,
                    "rf_annual": cfg.rf_annual,
                    "cost_per_trade": cfg.cost_per_trade,
                    "final_equity": float(eq.iloc[-1]),
                    **{f"ma_{k}": v for k, v in m.items()},
                    **{f"bh_{k}": v for k, v in bh_m.items()},
                }
            )

            # Walk-forward stability by time blocks
            bdf = metrics_by_blocks(eq, blocks)
            if not bdf.empty:
                bdf.insert(0, "run_tag", run_tag)
                bdf.insert(1, "ma_window", w)
                bdf.insert(2, "allocation", alloc)
                block_rows.append(bdf)

    summary = pd.DataFrame(summary_rows).sort_values(["ma_window", "allocation"])
    summary_path = OUT_DIR / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    if block_rows:
        blocks_df = pd.concat(block_rows, ignore_index=True)
        blocks_path = OUT_DIR / "walkforward_blocks_metrics.csv"
        blocks_df.to_csv(blocks_path, index=False)
        print(f"Saved walk-forward block metrics to {blocks_path}")
    else:
        blocks_df = pd.DataFrame()

    # -----------------------------
    # Parameter stability analysis (print table)
    # -----------------------------
    cols_to_show = [
        "ma_window", "allocation",
        "ma_cagr", "ma_ann_vol", "ma_sharpe_rf0", "ma_max_dd",
        "bh_cagr", "bh_ann_vol", "bh_sharpe_rf0", "bh_max_dd",
        "final_equity",
        "run_tag",
    ]
    display_df = summary[cols_to_show].copy()
    for c in ["ma_cagr", "ma_ann_vol", "ma_max_dd", "bh_cagr", "bh_ann_vol", "bh_max_dd"]:
        display_df[c] = display_df[c] * 100.0

    print("\n=== Parameter Stability Summary (percent where applicable) ===")
    print(display_df.to_string(index=False))

    # -----------------------------
    # Plots: best run vs buy&hold (best by Sharpe)
    # -----------------------------
    best_idx = summary["ma_sharpe_rf0"].idxmax()
    best = summary.loc[best_idx]
    best_tag = str(best["run_tag"])
    best_w = int(best["ma_window"])
    best_alloc = float(best["allocation"])
    print(f"\nBest by Sharpe: {best_tag} (MA={best_w}, alloc={best_alloc})")

    best_res = pd.read_parquet(OUT_DIR / f"res_{best_tag}.parquet")
    best_eq = best_res["equity"]
    bh_eq = bh_equity_full.loc[best_eq.index[0]: best_eq.index[-1]]

    plt.figure()
    best_eq.plot(label=f"Strategy ({best_tag})")
    bh_eq.plot(label="Buy & Hold")
    plt.title("Equity Curve Comparison (Best by Sharpe)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()

    plt.figure()
    dd(best_eq).plot(label=f"Strategy ({best_tag})")
    dd(bh_eq).plot(label="Buy & Hold")
    plt.title("Drawdown Comparison (Best by Sharpe)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.show()

    # Walk-forward plot: CAGR by block for best params
    if not blocks_df.empty:
        sel = blocks_df[blocks_df["run_tag"] == best_tag].copy()
        if not sel.empty:
            plt.figure()
            pd.Series(sel["cagr"].values, index=pd.to_datetime(sel["block_start"])).mul(100).plot(marker="o")
            plt.title(f"Walk-forward Block CAGR ({best_tag})")
            plt.xlabel("Block start")
            plt.ylabel("CAGR (%)")
            plt.show()


if __name__ == "__main__":
    main()
