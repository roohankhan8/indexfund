# Differences Between enhanced.ipynb and enhanced-v2.ipynb

## Overview
`enhanced-v2.ipynb` represents an improved version of the fund flow prediction pipeline with several critical methodological changes addressing data quality issues and model selection concerns identified in `enhanced.ipynb`.

---

## Key Differences

### 1. **Macro Data Handling & 2020 Rows**

#### `enhanced.ipynb`
```python
# Macro columns (oil, interest rate, USD) start from 2021.
# bfill fills pre-2021 rows with the earliest real value we have;
# ffill then closes any remaining gaps within the series.
enhanced_monthly[['oil_price', 'interest_rate', 'usd_exchange']] = (
    enhanced_monthly[['oil_price', 'interest_rate', 'usd_exchange']]
    .bfill()
    .ffill()
)
```
- **Keeps 2020 rows** with backward-filled macro data
- All 2020 macro values are **fabricated** (filled with earliest 2021 observations)
- Creates flat, unrealistic macro values for 2020

#### `enhanced-v2.ipynb`
```python
# Macro columns: ffill only (bfill removed — 2020 rows will be dropped below)
enhanced_monthly[['oil_price', 'interest_rate', 'usd_exchange']] = (
    enhanced_monthly[['oil_price', 'interest_rate', 'usd_exchange']]
    .ffill()
)

# Suggestion 1: Drop 2020 rows
# Macro data starts 2021; all 2020 macro values were forward-filled from the
# earliest 2021 observation — fabricated values for oil, interest rate, USD.
# Training on fabricated flat macro values corrupts the model.
enhanced_monthly = enhanced_monthly[
    enhanced_monthly['date'].dt.year >= 2021
].reset_index(drop=True)
```
- **Explicitly drops all 2020 rows**
- Prevents training on fabricated macro data
- Results in cleaner, more reliable training dataset (~50 rows instead of ~60)

**Impact**: V2 avoids training on corrupted data that could mislead the model about macro-fund flow relationships.

---

### 2. **Feature Engineering**

#### `enhanced.ipynb`
```python
# Lagged features
enhanced_monthly['lag_volume'] = enhanced_monthly['total_volume'].shift(1)
enhanced_monthly['lag_return'] = enhanced_monthly['log_return'].shift(1)
enhanced_monthly['lag_oil'] = enhanced_monthly['oil_price'].shift(1)
enhanced_monthly['lag_ir'] = enhanced_monthly['interest_rate'].shift(1)
enhanced_monthly['lag_usd'] = enhanced_monthly['usd_exchange'].shift(1)

features = ['lag_volume', 'lag_return', 'lag_oil', 'lag_ir', 'lag_usd']
```
- **5 features**: All lagged level values
- Assumes linear relationship with absolute levels

#### `enhanced-v2.ipynb`
```python
# Lagged level features
enhanced_monthly['lag_volume'] = enhanced_monthly['total_volume'].shift(1)
enhanced_monthly['lag_return'] = enhanced_monthly['log_return'].shift(1)
enhanced_monthly['lag_oil'] = enhanced_monthly['oil_price'].shift(1)
enhanced_monthly['lag_ir'] = enhanced_monthly['interest_rate'].shift(1)
enhanced_monthly['lag_usd'] = enhanced_monthly['usd_exchange'].shift(1)

# Suggestion 2: First-difference features
# Direction of IR and USD change is more stationary across macro regimes than levels.
# Compute diff first, then lag by 1 — only information available at t-1 used.
enhanced_monthly['ir_change'] = enhanced_monthly['interest_rate'].diff()
enhanced_monthly['usd_pct_change'] = enhanced_monthly['usd_exchange'].pct_change()
enhanced_monthly['lag_ir_change'] = enhanced_monthly['ir_change'].shift(1)
enhanced_monthly['lag_usd_pct_change'] = enhanced_monthly['usd_pct_change'].shift(1)

features = ['lag_volume', 'lag_return', 'lag_oil', 'lag_ir', 'lag_usd',
            'lag_ir_change', 'lag_usd_pct_change']
```
- **7 features**: Original 5 + 2 first-difference features
- Adds `lag_ir_change` (interest rate change) and `lag_usd_pct_change` (USD % change)
- **Rationale**: Changes in IR and USD are more stationary and informative than absolute levels across different macro regimes

**Impact**: V2 captures directional momentum in macro variables, improving prediction stability.

---

### 3. **Model Selection**

#### `enhanced.ipynb`
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
```
- **Random Forest**: Non-parametric ensemble method
- High variance risk with ~50 observations
- Prone to overfitting on small datasets
- No feature scaling required

#### `enhanced-v2.ipynb`
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Suggestion 3: Ridge replaces Random Forest
# Ridge + StandardScaler: low variance, handles correlated macro features,
# interpretable coefficients, no overfitting risk on ~50 rows.
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
```
- **Ridge Regression**: Linear model with L2 regularization
- Low variance, more stable with small datasets
- Handles correlated features (macro variables often correlate)
- Interpretable coefficients
- Includes `StandardScaler` for feature normalization

**Impact**: V2 uses a more appropriate model for the dataset size, reducing overfitting risk significantly.

---

### 4. **Prediction Interpretation**

#### `enhanced.ipynb`
- No explicit guidance on using predictions
- Treats predicted magnitudes as reliable

#### `enhanced-v2.ipynb`
```python
print(f'\nPredicted next fund flow: {next_pred[0]:.4f}')
# Suggestion 4: treat as directional signal only — not the magnitude
print('Note: use directional signal only — magnitude is unreliable at this dataset size')
```
- **Explicit warning**: Use predictions for direction (inflow/outflow) only
- Acknowledges magnitude estimates are unreliable with ~50 training samples

**Impact**: V2 sets realistic expectations for model utility.

---

### 5. **Evaluation Methodology**

#### `enhanced.ipynb`
- TimeSeriesSplit with 5 folds
- **Additional 80/20 train/test split** with R² and adj-R² metrics
- Final plot shows test set predictions

#### `enhanced-v2.ipynb`
- TimeSeriesSplit with 5 folds **only**
- No separate 80/20 split evaluation
- Focus on cross-validation RMSE across folds

**Impact**: V2 simplifies evaluation to focus on time-series cross-validation, avoiding redundant evaluations.

---

### 6. **Visualization Titles**

#### `enhanced.ipynb`
```python
fig.suptitle('Fund Flow Predictions — TimeSeriesSplit (5 Folds)', ...)
```

#### `enhanced-v2.ipynb`
```python
fig.suptitle('Fund Flow Predictions — Ridge + TimeSeriesSplit (5 Folds)', ...)
```
- V2 explicitly mentions Ridge model in title

---

## Summary Table

| Aspect | enhanced.ipynb | enhanced-v2.ipynb |
|--------|---------------|------------------|
| **2020 Rows** | Kept (with bfilled macro data) | Dropped (avoids fabricated data) |
| **Features** | 5 (lagged levels only) | 7 (levels + IR/USD changes) |
| **Model** | Random Forest | Ridge Regression + StandardScaler |
| **Overfitting Risk** | High (complex model, small data) | Low (regularized linear model) |
| **Prediction Use** | Magnitude estimates | Directional signals only |
| **Evaluation** | TimeSeriesSplit + 80/20 split | TimeSeriesSplit only |
| **Data Points** | ~57 after dropna | ~47 after dropping 2020 + dropna |

---

## Recommendation

**Use `enhanced-v2.ipynb`** for production/finalized analysis:
- More rigorous data quality checks (removes fabricated 2020 data)
- Better feature engineering (captures macro momentum)
- Appropriate model choice for dataset size (Ridge vs Random Forest)
- Realistic expectations on prediction reliability

`enhanced.ipynb` serves as the exploratory baseline that revealed issues addressed in V2.
