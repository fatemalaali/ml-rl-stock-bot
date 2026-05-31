# AAPL Stock Direction Prediction — ML & RL

Machine learning and reinforcement learning pipeline for predicting Apple Inc. (AAPL)
stock movement, built for IT9201 — Machine Learning and Data Mining.

Daily OHLCV data (2010–2026, ≈4000 trading days) is downloaded from Yahoo Finance,
enriched with technical indicators and cross-asset features, and used to train
classifiers that predict the **5-day forward price direction** (Up / Down). A tabular
Q-Learning agent is also trained to make buy/sell/hold trading decisions.

## Contents

| Path | Description |
|------|-------------|
| `stock_data_ml_rl.ipynb` | Main notebook — data acquisition, EDA, feature engineering, ML training, RL agent, and final comparison |
| `no-code.ows` | Orange workflow (no-code equivalent of the pipeline) |
| `data/` | Raw and cleaned AAPL datasets plus cross-asset context data |
| `figures/EDA/` | Exploratory data analysis plots |
| `figures/models/` | Model evaluation plots (confusion matrices, ROC, feature importance) |
| `figures/rl/` | Reinforcement learning training and portfolio plots |
| `models/` | Trained model artifacts (`.pkl`), the tuned Q-table (`.npy`), the scaler, and results summary CSVs |

## Pipeline overview

1. **Data acquisition** — download AAPL OHLCV from Yahoo Finance, save raw CSV.
2. **EDA** — distributions, correlations, class balance, candlestick/volume plots.
3. **Feature engineering** — momentum, trend, oscillator, volatility and microstructure
   indicators, plus six cross-asset features (S&P 500, VIX, 10Y yield, tech sector,
   relative strength). Target: `signal = 1` if `Close_{t+5} > Close_t`.
4. **ML models** — Random Forest, LightGBM, SVM, Logistic Regression, each tuned with
   `RandomizedSearchCV` (scored on balanced accuracy), combined in a soft-voting ensemble.
5. **Reinforcement learning** — tabular Q-Learning agent (RSI × MA-ratio state space,
   buy/sell/hold actions) optimising cumulative portfolio return.
6. **Final comparison** — Accuracy, F1, ROC-AUC across all models; RL evaluated by
   portfolio return.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook stock_data_ml_rl.ipynb
```

Then run the notebook top to bottom. Data acquisition requires an internet connection
(Yahoo Finance); the saved CSVs in `data/` allow offline reruns of later sections.

## Acknowledgement

Claude AI by Anthropic was used for code generation.
