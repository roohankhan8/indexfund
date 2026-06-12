# `3_final_model/` — complete file inventory + models + results

Central **final experimentation tree**: Jupyter notebooks **`enhanced-v1`→`enhanced-v8`**, reproducible **`scripts/`**, mirrored **`data/`**, generated **`output*`** diagnostics, explanatory markdowns (`explanations/`, `mds/`).

---

## Scripts (runnable summaries)

| File | Frequency / idea | Models | Outputs |
|------|------------------|--------|---------|
| `scripts/enhanced-v5.py` | **Monthly** (and monthly-style feature lags/vol/abnormal volume/macros+CPI themes per that iteration) | **Ridge**, **ElasticNet**, **GradientBoostingRegressor**, **XGBoostRegressor**, **LightGBM** (if installed) per script configuration | Saves numbered plots under `scripts/../output/` (see `SAVE_PLOTS` / `OUT_DIR` in-file) |
| `scripts/enhanced-v7.py` | **Weekly** resampling + technical indicators built from **`kse30_index_level.csv`** | Same sklearn model families as above, trained on weekly rows | Saves numbered PNGs under `scripts/../output/`; includes longer-horizon forecast figures per script tail sections |

Prefer re-running scripts over stale PNGs when numbers must match prose.

---

## Notebooks (research lineage)

| Notebook | Narrative milestone |
|-----------|---------------------|
| `enhanced-v1.ipynb` | Baseline pipeline; RF overfitting pitfalls on ~57 rows documented in archived `mds/plan` lineage sections |
| `enhanced-v2.ipynb` | Data fixes (drop unreliable 2020 macro fabrications), ridge baseline |
| `enhanced-v3.ipynb` | Multi-model + directional metric; naive LSTM attempt (shown ineffective) |
| `enhanced-v4.ipynb` | CPI + iterative multi-step forecasting attempts; regressions enumerated in bugs list |
| `enhanced-v5.ipynb` | Stabilised feature stack + TSCV grids + 12-feature rationale |
| `enhanced-v6.ipynb` | Bridge iteration between v5 monthly & v7 weekly ideas (see paired `output-v6` charts) |
| `enhanced-v7.ipynb` | Weekly + technical-indicator experiments |
| `enhanced-v8.ipynb` | Further polish / speculative adjustments (consult notebook intro cells) |
| `grok/grok-notebook-v1.ipynb` | Synthetic/abnormal-volume flow proxy investigations + Hurst/VR exploratory studies |
| `grok/grok-notebook-v2.ipynb` | Per-fund normalization + corr heatmaps; known merge bug flagged historically |

---

## Local data (`data/`)

| File | Role |
|------|------|
| `kse-30-basic.xlsx` | Panel: tickers × day with price / weight % / liquidity |
| `kse30_index_level.csv` | Pre-aggregated **index-level** returns & volumes (**preferred** equity signal source) |
| `funds_data.xlsx` | Sheets `AKD`, `NBP`, `NTI` with NAV+AUM histories |
| `macro_data.xlsx` | Tabs `OIL`, `USD`, `IR` aligned macro drivers |
| `cpi.csv` | Monthly CPI YoY ingest |

---

## Documentation / meta (`mds/` + `explanations/`)

| File | Contents |
|------|----------|
| `mds/context.md` | Early pipeline narrated walkthrough |
| `mds/bugs.md` | Logic traps found in notebook v4 (return calc, iterative forecast stubs, naive LSTM timestep) |
| `mds/plan.md` | Consolidated plan + summarized outcomes + limitation essay |
| `mds/diff.md`, `mds/extra.md` | Comparative notes between iterations |
| `explanations/v5-explanation.md` | Interpretation **per chart number** (`01_*` … `09_*`) for monthly diagnostics |
| `explanations/v7-explanation.md` | Mirrors for weekly enhancement narrative |

---

## Generated figures (`output/` baseline)

| PNG | Plot meaning |
|-----|----------------|
| `01_correlation_heatmap.png` | Feature ↔ feature + vs target correlations |
| `02_fund_flow_history.png` | Winsorised flow histogram / bar chronology |
| `03_tscv_folds.png` | TSCV overlays per model × fold grid |
| `04_holdout_predictions.png` | Holdout predictive curves |
| `05_model_comparison.png` | RMSE / directional duel bar charts |
| `06_feature_importance.png` | Tree-importance dashboards |
| `07_linear_coefficients.png` | Ridge / Elastic standardized coefficients |
| `08_forecast_12week.png` | Tactical multi-step tactical forecast framing |
| `09_company_selection.png` | Top constituents selection logic |
| `10_forecast_2year.png` | Long horizon stress illustration (scenario not point-truth) |

## Version overlays (every PNG currently in repo)

### `output-v5/`

- `01_correlation_heatmap.png`
- `02_fund_flow_history.png`
- `03_tscv_folds.png`
- `04_holdout_predictions.png`
- `05_model_comparison.png`
- `06_feature_importance.png`
- `07_linear_coefficients.png`
- `08_forecast_12week.png`
- `09_company_selection.png`
- `10_forecast_2year.png`

### `output-v6/` *(9 images; no `10_forecast_2year` in repo)*

- `01_correlation_heatmap.png`
- `02_fund_flow_history.png`
- `03_tscv_folds.png`
- `04_holdout_predictions.png`
- `05_model_comparison.png`
- `06_feature_importance.png`
- `07_linear_coefficients.png`
- `08_forecast_6month.png`
- `09_company_selection.png`

### `output-v7/`

- `01_correlation_heatmap.png`
- `02_fund_flow_history.png`
- `03_tscv_folds.png`
- `04_holdout_predictions.png`
- `05_model_comparison.png`
- `06_feature_importance.png`
- `07_linear_coefficients.png`
- `08_forecast_12week.png`
- `09_company_selection.png`
- `10_forecast_2year.png`

### `output/` (unversioned/default bundle)

Mirrors the same numbering pattern as **`output-v5/`**: `01`–`10` files listed above (`08_forecast_12week.png`, `09_company_selection.png`, `10_forecast_2year.png`, etc.).

## Models used (conceptual rollup)

Across versions you touched:

| Family | Algorithms | Typical purpose |
|--------|-------------|------------------|
| Penalised linear | **Ridge**, **ElasticNet** | Interpretable coefficients; guard against \(p \gg n\) instability |
| Shallow ensembles | **Sklearn HistGradient boosting path / GradientBoostingRegressor**, **XGBoostRegressor**, **LightGBM** | Non-linear lift attempt (watch overfit) |
| Discarded/skeptical | **RandomForest (early)**, **LSTM/1-step window** | Shown brittle or academically indefensible given sample |

---

## Key quantitative results (`enhanced-v5` monthly diagnostics)

_Source: `3_final_model/explanations/v5-explanation.md`. Holdout slice ≈10 months — interpret cautiously._

| Metric / insight | Observation |
|------------------|----------------|
| Target vs feature correlations | All \|corr\| ≤ **0.19** → inherently low linear explainability |
| Holdout \(R^2\) (trees) | **Negative** for heavy boosting ⇒ worse than unconditional mean predictor |
| Holdout \(R^2\) (Ridge ~0.04, ElasticNet ~0.02) | Only marginally better than naive mean |
| Directional Accuracy | +/-10% swings on 10 observations not statistically decisive |
| Rare-month spike miss | Largest negative flow month missed by ≥500 PKR equivalents even after ±3σ winsorization clipping |
| Company selection caveat (pre-fix) | Un-winsorized returns let single ticker dominate ranking (fixed ideology in weekly v7 code path) |

**Weekly v7** (`explanations/v7-explanation.md`): increases effective sample (>200 periods) stabilizes TSCV folds, adds technical signals; revisit printed metrics in notebook output cells / regenerated PNG overlays.

---

## How to rerun

```powershell
.venv\Scripts\activate
python 3_final_model/scripts/enhanced-v7.py
# or notebook counterpart
jupyter notebook 3_final_model/enhanced-v5.ipynb
```

Outputs land under `output*/` beside the notebook’s configured `SAVE_PLOTS` path.
