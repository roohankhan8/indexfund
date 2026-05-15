# 8_last_model Workflow Notes

## Objective
Build an end-to-end monthly fund-flow prediction and portfolio-tilt workflow for KSE-30 index funds (AKD, NBP, NTI) using market and macro lagged signals.

## Run
```powershell
python pipeline.py
```

## Inputs
- `data/funds_data.xlsx` (sheets: AKD, NBP, NTI)
- `data/kse30_daily_data.csv`
- `data/macro_data.xlsx` (OIL, IR, USD)
- `data/cpi.csv`

## Outputs
- `output/tables/model_frame_monthly.csv`
- `output/tables/test_predictions.csv`
- `output/tables/strategy_backtest.csv`
- `output/metrics/metrics_summary.json`
- `output/figures/strategy_cumulative_return.png`

## Method summary
1. Compute monthly flow using: `flow_t = AUM_t - AUM_(t-1) * (NAV_t / NAV_(t-1))`
2. Build monthly market features from KSE-30 constituents.
3. Build monthly macro features with forward-filled level series and transformed changes.
4. Create lagged predictors (1,2,3 months).
5. Forecast next-month fund flows and directions (RandomForest regressor/classifier).
6. Convert predicted flows into positive-score portfolio tilts; compare with equal-weight baseline.
