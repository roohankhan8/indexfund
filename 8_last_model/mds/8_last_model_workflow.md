# 8_last_model Workflow Notes

## Primary Objective
Predict for each current KSE-30 company at rebalance date:
1. Whether it will **stay** in KSE-30 at next rebalance (or be excluded)
2. Expected **change in index weight** at next rebalance

NAV/AUM data from AKD, NBP, and NTI funds are mandatory model features.

## Main Script
```powershell
python kse30_rebalance_pipeline.py
```

## Inputs
- `data/kse30_daily_data.csv`
- `data/funds_data.xlsx` (AKD, NBP, NTI)

## Target Construction
- Rebalance snapshots are taken at **March and September month-end**.
- For each symbol in snapshot `t`:
  - `stay_next = 1` if symbol appears in snapshot `t+1`, else `0`
  - `weight_change_next = next_weight - current_weight` (next weight is `0` if excluded)

## Feature Groups
- Symbol-level market features:
  - current index weight and rank
  - 21-day return proxy
  - 63-day volatility and turnover proxies
- Fund features (AKD/NBP/NTI):
  - NAV, AUM, and flow (current + lags)
  - totals across funds
  - derived ratios (`aum_nav_ratio`, `flow_to_aum`)

## Outputs
- `output/tables/kse30_rebalance_training_panel.csv`
- `output/tables/kse30_rebalance_test_predictions.csv`
- `output/tables/kse30_next_rebalance_forecast.csv`
- `output/metrics/kse30_rebalance_metrics.json`
- `output/figures/kse30_stay_rate_actual_vs_pred.png`
