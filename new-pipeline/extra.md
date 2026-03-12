## Summary of Analysis

### Notebook Progression

| Notebook | Key Contribution | Key Defect |
|---|---|---|
| **enhanced-v1** | Baseline pipeline, Winsorization, TSCV | bfill fabricates 2020 macro values; RF overfits |
| **enhanced-v2** | Drops 2020 rows, Ridge model, first-diff macro features | TSCV only, no directional accuracy |
| **enhanced-v3** | Multi-model (Ridge/XGB/LSTM), directional accuracy metric | Reverted 2020 row drop; 1-timestep LSTM |
| **grok-v1** | Synthetic flow proxy, Hurst Exponent, avg_weight_change feature | No real fund data |
| **grok-v2** | AUM-normalized flows, correlation heatmap, multi-lag features | Runtime error in portfolio cell |
| **enhanced-v4** | CPI feature, 6-month forecast, auto model selection | LSTM fails (RMSE 188k vs Ridge 449); 2020 data back; first-diff features dropped |

### What v5 Will Do Differently

1. **`kse30_index_level.csv`** as primary KSE-30 source — already has proper `log_return` and `avg_weight_change` (fixes the weighted_price-as-return bug)
2. **12 features** including multi-lag (1+3 month), return volatility, abnormal volume, CPI, and first-diff macros
3. **4 models**: ElasticNet (new), Ridge, LightGBM, GradientBoosting — no LSTM
4. **TimeSeriesSplit(5)** + directional accuracy as the primary evaluation metric
5. **6-month forecast** with all macros held constant (documented assumption)

### Open Questions for You

1. **`.py` script or `.ipynb` notebook?** — plan says `.py`; confirm?
2. **Plots**: save to PNG files or `plt.show()` interactively?
3. **LightGBM**: installed in your venv, or use sklearn `GradientBoostingRegressor` instead?

Made changes.