# src/ml_walkforward_train.py
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

INP = Path("data/features/SPY_ml_dataset.parquet")
OUT = Path("backtest/ml_results_regime_on.parquet")

TRAIN_YEARS = 10
TEST_YEARS = 3

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_parquet(INP).sort_index()

features = [
    # Returns
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",

    # Volatility
    "vol_20d", "vol_60d",

    # Trend / regime
    "dist_ma200",
    "ma_slope",
    "ma50_200",

    # Momentum
    "mom_3m",
    "mom_6m",

    # Drawdown
    "drawdown",

    "trend_6m",
    "trend_12m",


    # Volatility regime
    #"vol_ratio",
    #"vol_z",

]


# Regime filter: only train/predict in MA200 risk-on regime
df = df[df["regime_ma200"] == 1].copy()

X = df[features]
y = df["target"].astype(float)  # continuous forward return
dates = df.index

# -----------------------------
# Walk-forward splits
# -----------------------------
def make_splits(dates_index: pd.DatetimeIndex, train_years: int, test_years: int):
    splits = []
    start = dates_index.min()
    end = dates_index.max()

    cur = pd.Timestamp(start.year, start.month, start.day)

    while True:
        train_end = cur + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        train_idx = (dates_index >= cur) & (dates_index < train_end)
        test_idx = (dates_index >= train_end) & (dates_index < test_end)

        if train_idx.sum() > 800 and test_idx.sum() > 150:
            splits.append((train_idx, test_idx, train_end, test_end))

        cur = cur + pd.DateOffset(years=test_years)

    return splits


splits = make_splits(dates, TRAIN_YEARS, TEST_YEARS)
print("Number of splits:", len(splits))

# -----------------------------
# Model (REGRESSION)
# -----------------------------
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)

# -----------------------------
# Walk-forward training
# -----------------------------
all_pred = []
all_true = []
all_dates = []

for i, (train_idx, test_idx, train_end, test_end) in enumerate(splits):
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    # Also compute correlation (often more informative than R^2 in finance)
    corr = np.corrcoef(y_test.values, pred)[0, 1] if len(y_test) > 2 else np.nan

    print(f"Split {i} ({train_end.date()}→{test_end.date()}): RMSE={rmse:.4f}, R2={r2:.4f}, Corr={corr:.4f}")

    all_pred.append(pred)
    all_true.append(y_test.values)
    all_dates.append(dates[test_idx].values)

print("\nAggregating results...")

pred = np.concatenate(all_pred) if all_pred else np.array([])
true = np.concatenate(all_true) if all_true else np.array([])
test_dates = np.concatenate(all_dates) if all_dates else np.array([])

if len(true) == 0:
    raise RuntimeError("No test predictions produced. Check your split logic after regime filtering.")

overall_rmse = np.sqrt(mean_squared_error(true, pred))
overall_r2 = r2_score(true, pred)
overall_corr = np.corrcoef(true, pred)[0, 1] if len(true) > 2 else np.nan

print("\n=== Overall ML Performance (REGRESSION, Regime=MA200 risk-on only) ===")
print("RMSE:", round(float(overall_rmse), 6))
print("R2:", round(float(overall_r2), 6))
print("Corr:", round(float(overall_corr), 6))

res = pd.DataFrame(
    {
        "date": test_dates,
        "pred_ret_fwd": pred,   # predicted forward return over HORIZON days
        "true_ret_fwd": true,
    }
).set_index("date").sort_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_parquet(OUT)

print("Saved regression predictions:", OUT)
print("Rows saved:", len(res))
print(res.tail())
