## General Summary:
This trading strategy combinds a systemic trading with machine Learning
predictions and traditional market regime filtering to trade the SPY.
The goal of this project was to replicate the "Golden path" pipeline
that turns raw price data into a predictive dataset, this is achieved 
by using 60-day forward returns as the model target.

## Setup:
source .venv/bin/activate (ALWAYS THIS)
pip install -r requirements.txt

## Data:
python src/download_spy.py

## Clean:
python src/clean_spy.py

## Features:
python src/features_basic.py

## Plots:
python src/plot_spy_sanity.py
python src/inspect_returns.py

## Machine Learning Pipeline:

After verifying the data with plots, run the label engineering and model training scripts:

1. Generate Target Labels:
   python src/ml_label_engineering.py
   - Calculates 60-day forward returns [2].
   - Saves final training set to data/features/SPY_ml_dataset.parquet [3].

2. Train Walk Forward Model:
   python src/ml_walkforward_train.py
   - Uses a Ridge regression pipeline with a 10-year training and 3-year testing window [4].
   - Only trains and predicts during MA200 "Risk-On" regimes [5].
   - Saves predictions to backtest/ml_results_regime_on.parquet [4].

## Backtesting Strategy:

Run the final strategy script to evaluate the Conservative Rank-ML performance:

python src/backtest_rank_ml_ma200.py
- Implements the Rank-to-Weight mapping: weight = 0.5 + K * (rank - 0.5) [6].
- Targets 6% annual volatility with a 20-day lookback [6, 7].
- Saves performance metrics to backtest/rank_ml_ma200_results.parquet [8, 9].

## Strategy Parameters (Conservative Run):

The successful results in this repository were achieved using the following "Golden Path" parameters:
- Horizon: 60-day forward returns [2].
- K Value: 1.00 (Mapping sensitivity) [6, 7].
- Target Volatility: 6.00% [6, 7].
- Max Leverage: 1.50 cap [6].
- Regime Filter: Price > MA200 [5, 10].

## Final Performance Metrics:

- Total Return: 37.76% [Source: Conservative Rank-ML Results].
- CAGR: 1.23% [Source: Conservative Rank-ML Results].
- Sharpe Ratio: 0.50 [Source: Conservative Rank-ML Results].
- Max Drawdown: -5.15% [Source: Conservative Rank-ML Results].
- Total Trades: 111 (entries + exits) [Source: Conservative