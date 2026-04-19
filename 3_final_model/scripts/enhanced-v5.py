# =============================================================================
# enhanced-v5.py — Fund Flow Prediction Pipeline
# Best techniques distilled from enhanced-v1 through v4 + grok-v1/v2
# =============================================================================
#
# PACKAGES TO INSTALL (only if missing from your .venv):
#   pip install lightgbm          ← optional; code falls back to XGBoost without it
#   pip install PyPortfolioOpt    ← optional; only needed for Section 9 portfolio opt
#
# Run from project root:
#   .venv\Scripts\python new-pipeline/v5/enhanced-v5.py
# =============================================================================

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
N_SPLITS        = 5
FORECAST_MONTHS = 6
SAVE_PLOTS      = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
OUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def save_fig(name: str):
    if SAVE_PLOTS:
        path = os.path.join(OUT_DIR, f'{name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"    → saved: {path}")
    plt.show()


def excel_to_datetime(serial):
    try:
        return datetime(1899, 12, 30) + timedelta(days=int(serial))
    except (ValueError, TypeError):
        return pd.NaT


def parse_date_col(series: pd.Series) -> pd.Series:
    """Handle both Excel serial numbers and string date formats."""
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(excel_to_datetime)
    return pd.to_datetime(series, errors='coerce')


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """R/S analysis to test market persistence (H > 0.5 = persistent)."""
    lags = range(2, min(max_lag, len(ts) // 2))
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])


def variance_ratio(ts: np.ndarray, q: int = 3) -> float:
    """Variance ratio test (VR = 1 → random walk, VR > 1 → positive autocorr)."""
    mu   = np.mean(ts)
    var1 = np.var(ts - mu, ddof=1)
    ts_q = [ts[i] - ts[i - q] - q * mu for i in range(q, len(ts))]
    return float(np.var(ts_q, ddof=1) / (q * var1))


# ─── STEP 1: KSE-30 INDEX DATA ───────────────────────────────────────────────
print("\n[STEP 1] Loading KSE-30 index level data (kse30_index_level.csv)...")
# Using pre-computed daily returns — fixes the weighted_price proxy bug from v1-v4
kse_daily = pd.read_csv(
    os.path.join(DATA_DIR, 'kse30_index_level.csv'),
    parse_dates=['date']
)
kse_daily = kse_daily.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

kse_monthly = (
    kse_daily.set_index('date').resample('ME').agg(
        log_return        = ('log_return', 'sum'),    # sum of daily logs = monthly log return
        total_volume      = ('total_volume', 'sum'),
        avg_weight_change = ('avg_weight_change', 'mean'),
    ).reset_index()
)

# Derived rolling features (computed before lagging — no data leakage)
kse_monthly['vol3_return']  = kse_monthly['log_return'].rolling(3).std()     # return volatility
kse_monthly['avg3m_volume'] = kse_monthly['total_volume'].rolling(3).mean()
kse_monthly['abnorm_vol']   = kse_monthly['total_volume'] / kse_monthly['avg3m_volume']  # relative activity

print(f"    Rows: {len(kse_monthly)}  |  "
      f"{kse_monthly['date'].min().date()} → {kse_monthly['date'].max().date()}")


# ─── STEP 2: COMPANY DATA (portfolio selection only) ─────────────────────────
print("\n[STEP 2] Loading company-level data (kse-30-basic.xlsx)...")
co_raw = pd.read_excel(os.path.join(DATA_DIR, 'kse-30-basic.xlsx'))
co_raw.columns = co_raw.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
co_raw = co_raw.rename(columns={'idx_wt_%': 'idx_wt', 'date': 'date_raw'})
co_raw['date'] = parse_date_col(co_raw['date_raw'])
co_raw = (co_raw
          .dropna(subset=['date', 'company', 'price'])
          .drop_duplicates(subset=['date', 'company'])
          .sort_values(['company', 'date']))

co_raw['month'] = co_raw['date'].dt.to_period('M')

company_monthly = (
    co_raw.groupby(['company', 'month'])['price']
    .last()
    .reset_index()
)
company_monthly['return'] = company_monthly.groupby('company')['price'].pct_change()
company_monthly['month_dt'] = company_monthly['month'].dt.to_timestamp('ME')

print(f"    {co_raw['company'].nunique()} companies  |  "
      f"{co_raw['date'].min().date()} → {co_raw['date'].max().date()}")


# ─── STEP 3: FUND FLOW DATA ───────────────────────────────────────────────────
print("\n[STEP 3] Computing fund flows (AKD, NBP, NTI)...")
flow_frames = []
for fund_name in ['AKD', 'NBP', 'NTI']:
    fdf = pd.read_excel(os.path.join(DATA_DIR, 'funds_data.xlsx'), sheet_name=fund_name)
    fdf.columns = fdf.columns.str.strip().str.upper()
    fdf['date'] = parse_date_col(fdf['DATE'])
    fdf = fdf.dropna(subset=['date', 'NAV', 'AUM']).sort_values('date')
    fdf['nav_ratio'] = fdf['NAV'] / fdf['NAV'].shift(1)
    fdf['flow']      = fdf['AUM'] - (fdf['AUM'].shift(1) * fdf['nav_ratio'])
    flow_frames.append(fdf[['date', 'flow']])
    print(f"    {fund_name}: {len(fdf)} daily rows")

monthly_flow = (
    pd.concat(flow_frames, ignore_index=True)
    .set_index('date').resample('ME')['flow']
    .sum().reset_index(name='total_fund_flow')
)
# Fill only the first-row NaN (no prior period) with 0; rest is real data
monthly_flow['total_fund_flow'] = monthly_flow['total_fund_flow'].fillna(0)


# ─── STEP 4: MACRO DATA ───────────────────────────────────────────────────────
print("\n[STEP 4] Processing macro data (OIL, IR, USD)...")

# Oil — monthly close
oil = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='OIL')
oil['date'] = parse_date_col(oil['DATE'])
oil_monthly = (oil.dropna(subset=['date'])
               .set_index('date').resample('ME')['PRICE'].last()
               .reset_index().rename(columns={'PRICE': 'oil_price'}))

# Interest Rate — step function; forward fill daily before resampling
ir = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='IR')
ir['date'] = parse_date_col(ir['DATE'])
ir = ir.dropna(subset=['date']).set_index('date').sort_index()
ir_daily   = ir.reindex(pd.date_range(ir.index.min(), ir.index.max(), freq='D')).ffill()
ir_monthly = (ir_daily.resample('ME').last()
              .reset_index().rename(columns={'index': 'date', 'RATE': 'interest_rate'}))
ir_monthly.columns = ['date', 'interest_rate']
print(f"    IR coverage: {ir_monthly['date'].min().date()} → {ir_monthly['date'].max().date()}")

# USD — monthly close
usd = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='USD')
usd['date'] = parse_date_col(usd['DATE'])
usd_monthly = (usd.dropna(subset=['date'])
                .set_index('date').resample('ME')['USD'].last()
                .reset_index().rename(columns={'USD': 'usd_exchange'}))


# ─── STEP 5: CPI DATA ─────────────────────────────────────────────────────────
print("\n[STEP 5] Loading CPI data (cpi.csv)...")
cpi_raw = pd.read_csv(
    os.path.join(DATA_DIR, 'cpi.csv'),
    skiprows=1, header=None, names=['period', 'yoy_cpi']
)
cpi_raw = cpi_raw[cpi_raw['period'] != 'Period'].dropna()
cpi_raw['date']    = pd.to_datetime('01-' + cpi_raw['period'].astype(str),
                                    format='%d-%b-%y', errors='coerce')
cpi_raw['yoy_cpi'] = pd.to_numeric(cpi_raw['yoy_cpi'], errors='coerce')
cpi_monthly = (cpi_raw.dropna(subset=['date', 'yoy_cpi'])
               .set_index('date').resample('ME')['yoy_cpi'].last()
               .reset_index().rename(columns={'yoy_cpi': 'cpi'}))
print(f"    CPI coverage: {cpi_monthly['date'].min().date()} → {cpi_monthly['date'].max().date()}")


# ─── STEP 6: MERGE & CLEAN ────────────────────────────────────────────────────
print("\n[STEP 6] Merging all sources on monthly date...")
df = kse_monthly.copy()
for src in [monthly_flow, oil_monthly, ir_monthly, usd_monthly, cpi_monthly]:
    df = df.merge(src, on='date', how='left')

# Forward fill macros only (no bfill — prevents fabricating pre-2021 values)
for col in ['oil_price', 'interest_rate', 'usd_exchange', 'cpi']:
    df[col] = df[col].ffill()

# Fund flow: fill NaN with 0 (fund not yet active = no flow)
df['total_fund_flow'] = df['total_fund_flow'].fillna(0)

print(f"    Merged rows: {len(df)}")
print("    NaN counts (key columns):")
check_cols = ['total_fund_flow', 'oil_price', 'interest_rate', 'usd_exchange', 'cpi',
              'log_return', 'total_volume', 'vol3_return', 'abnorm_vol']
print("   ", df[check_cols].isnull().sum().to_string())


# ─── STEP 7: FILTER & WINSORIZE ──────────────────────────────────────────────
print("\n[STEP 7] Dropping pre-2021 rows and winsorizing fund flows...")
# Drop 2020 rows — macro data starts 2021; bfill would fabricate values (v2 fix)
df = df[df['date'].dt.year >= 2021].reset_index(drop=True)
print(f"    After 2020 drop: {len(df)} rows "
      f"({df['date'].min().date()} → {df['date'].max().date()})")

mean_flow = df['total_fund_flow'].mean()
std_flow  = df['total_fund_flow'].std()
lower_w   = mean_flow - 3 * std_flow
upper_w   = mean_flow + 3 * std_flow
n_clip    = int(((df['total_fund_flow'] < lower_w) | (df['total_fund_flow'] > upper_w)).sum())
df['total_fund_flow'] = df['total_fund_flow'].clip(lower=lower_w, upper=upper_w)
print(f"    Winsorized {n_clip} outlier rows  [±3σ: {lower_w:.0f}, {upper_w:.0f}]")


# ─── STEP 8: FEATURE ENGINEERING ─────────────────────────────────────────────
print("\n[STEP 8] Building features...")

# Lagged level features
df['lag1_volume']     = df['total_volume'].shift(1)
df['lag3_volume']     = df['total_volume'].shift(3)
df['lag1_return']     = df['log_return'].shift(1)
df['lag3_return']     = df['log_return'].shift(3)
df['lag1_vol3_ret']   = df['vol3_return'].shift(1)       # 3-month return volatility
df['lag1_abnorm_vol'] = df['abnorm_vol'].shift(1)        # abnormal trading activity

# Macro levels (lagged)
df['lag1_ir']  = df['interest_rate'].shift(1)
df['lag1_oil'] = df['oil_price'].shift(1)
df['lag1_usd'] = df['usd_exchange'].shift(1)
df['lag1_cpi'] = df['cpi'].shift(1)

# First-difference macro features — direction more stationary than level (v2 insight)
df['lag1_ir_change'] = df['interest_rate'].diff().shift(1)
df['lag1_usd_pct']   = df['usd_exchange'].pct_change().shift(1)

FEATURES = [
    'lag1_volume',    'lag3_volume',
    'lag1_return',    'lag3_return',
    'lag1_vol3_ret',  'lag1_abnorm_vol',
    'lag1_ir',        'lag1_ir_change',
    'lag1_oil',
    'lag1_usd',       'lag1_usd_pct',
    'lag1_cpi',
]
TARGET = 'total_fund_flow'

model_df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
print(f"    Feature matrix: {model_df.shape[0]} rows × {len(FEATURES)} features")
print(f"    Usable window:  {model_df['date'].min().date()} → {model_df['date'].max().date()}")

X = model_df[FEATURES].values
y = model_df[TARGET].values


# ─── PLOT 1: Correlation Heatmap ─────────────────────────────────────────────
print("\n[PLOT 1] Correlation heatmap (features vs target)...")
fig, ax = plt.subplots(figsize=(13, 9))
corr_cols  = FEATURES + [TARGET]
corr_labels = FEATURES + ['fund_flow']
corr_data  = model_df[corr_cols].copy()
corr_data.columns = corr_labels
corr_matrix = corr_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.75})
ax.set_title('Feature Correlation Matrix — enhanced-v5', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('01_correlation_heatmap')


# ─── PLOT 2: Fund Flow History ────────────────────────────────────────────────
print("\n[PLOT 2] Fund flow history...")
fig, ax = plt.subplots(figsize=(14, 4))
colors_bar = np.where(model_df[TARGET] >= 0, '#2ecc71', '#e74c3c')
ax.bar(model_df['date'], model_df[TARGET], color=colors_bar, edgecolor='none', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(lower_w, color='darkorange', linewidth=1.2, linestyle='--', label=f'−3σ = {lower_w:.0f}')
ax.axhline(upper_w, color='darkorange', linewidth=1.2, linestyle='--', label=f'+3σ = {upper_w:.0f}')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')
ax.set_title('Monthly Fund Flow AKD + NBP + NTI (Winsorized at ±3σ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fund Flow (PKR)')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('02_fund_flow_history')


# ─── STEP 9: MODEL DEFINITIONS ───────────────────────────────────────────────
print("\n[STEP 9] Initialising models...")

models: dict = {
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE)),
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1.0)),
    ]),
    'GradBoost': GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbosity=0,
    ),
}

if HAS_LGB:
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=100, num_leaves=15, min_child_samples=5,
        reg_alpha=1.0, reg_lambda=1.0, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=-1,
    )

MODEL_NAMES  = list(models.keys())
MODEL_COLORS = ['#e74c3c', '#3498db', '#9b59b6', '#27ae60', '#f39c12']
print(f"    Active models: {MODEL_NAMES}")


# ─── STEP 10: TIMESERIES CROSS-VALIDATION ────────────────────────────────────
print(f"\n[STEP 10] TimeSeriesSplit ({N_SPLITS} folds)...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_results = {name: {'rmse': [], 'mae': [], 'dir_acc': []} for name in MODEL_NAMES}
fold_preds  = {name: [] for name in MODEL_NAMES}   # (dates, y_true, y_pred) per fold

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dates_val   = model_df['date'].iloc[val_idx].values

    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_val)
        cv_results[name]['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_results[name]['mae'].append(mean_absolute_error(y_val, preds))
        cv_results[name]['dir_acc'].append(directional_accuracy(y_val, preds))
        fold_preds[name].append((dates_val, y_val, preds))

print()
print("  ╔══════════════════╦═════════════╦═════════════╦══════════════╗")
print("  ║ Model            ║  Mean RMSE  ║  Mean MAE   ║  Dir Acc %   ║")
print("  ╠══════════════════╬═════════════╬═════════════╬══════════════╣")
for name in MODEL_NAMES:
    mr = np.mean(cv_results[name]['rmse'])
    mm = np.mean(cv_results[name]['mae'])
    md = np.mean(cv_results[name]['dir_acc'])
    print(f"  ║ {name:<16} ║  {mr:>9.1f}  ║  {mm:>9.1f}  ║  {md:>10.1f}  ║")
print("  ╚══════════════════╩═════════════╩═════════════╩══════════════╝")


# ─── PLOT 3: CV Fold Results ──────────────────────────────────────────────────
n_m  = len(MODEL_NAMES)
fig, axes = plt.subplots(n_m, N_SPLITS, figsize=(18, 3.2 * n_m), sharey=True)
for mi, name in enumerate(MODEL_NAMES):
    for fi, (dv, yv, pv) in enumerate(fold_preds[name]):
        ax = axes[mi][fi]
        ax.plot(dv, yv, 'o-', color='steelblue',  markersize=4, linewidth=1.2, label='Actual')
        ax.plot(dv, pv, 's--', color='tomato',    markersize=4, linewidth=1.2, label='Predicted')
        ax.axhline(0, color='gray', linewidth=0.5)
        if mi == 0:
            ax.set_title(f'Fold {fi + 1}', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        if fi == 0:
            ax.set_ylabel(name, fontsize=9, fontweight='bold')
        if mi == 0 and fi == N_SPLITS - 1:
            ax.legend(fontsize=7)
fig.suptitle(f'TimeSeriesSplit ({N_SPLITS} Folds) — Actual vs Predicted per Model',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('03_tscv_folds')


# ─── STEP 11: HOLDOUT EVALUATION (last 20%) ──────────────────────────────────
print("\n[STEP 11] Holdout evaluation (last 20%)...")
split_idx = int(len(X) * 0.8)
X_tr_h, X_te_h = X[:split_idx], X[split_idx:]
y_tr_h, y_te_h = y[:split_idx], y[split_idx:]
dates_te        = model_df['date'].iloc[split_idx:].values

holdout_preds:   dict = {}
holdout_metrics: dict = {}

for name, mdl in models.items():
    mdl.fit(X_tr_h, y_tr_h)
    preds = mdl.predict(X_te_h)
    holdout_preds[name] = preds
    holdout_metrics[name] = {
        'RMSE':   float(np.sqrt(mean_squared_error(y_te_h, preds))),
        'MAE':    float(mean_absolute_error(y_te_h, preds)),
        'R2':     float(r2_score(y_te_h, preds)),
        'DirAcc': directional_accuracy(y_te_h, preds),
    }

print()
print("  ╔══════════════════╦══════════╦══════════╦══════════╦══════════╗")
print("  ║ Model            ║   RMSE   ║   MAE    ║    R²    ║ DirAcc % ║")
print("  ╠══════════════════╬══════════╬══════════╬══════════╬══════════╣")
for name, m in holdout_metrics.items():
    print(f"  ║ {name:<16} ║ {m['RMSE']:>8.1f} ║ {m['MAE']:>8.1f} ║ "
          f"{m['R2']:>8.3f} ║ {m['DirAcc']:>8.1f} ║")
print("  ╚══════════════════╩══════════╩══════════╩══════════╩══════════╝")


# ─── PLOT 4: Holdout Predictions ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_te, y_te_h, 'o-', color='steelblue', linewidth=2,
        markersize=5, label='Actual', zorder=5)
for (name, preds), color in zip(holdout_preds.items(), MODEL_COLORS):
    r2 = holdout_metrics[name]['R2']
    da = holdout_metrics[name]['DirAcc']
    ax.plot(dates_te, preds, '--', color=color, linewidth=1.5, markersize=4,
            label=f'{name}  (R²={r2:.2f}, DA={da:.0f}%)')
ax.axhline(0, color='gray', linewidth=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, ha='right')
ax.set_title('Holdout Predictions — All Models vs Actual', fontsize=12, fontweight='bold')
ax.set_ylabel('Fund Flow (PKR)')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('04_holdout_predictions')


# ─── PLOT 5: Model Comparison Bar Charts ─────────────────────────────────────
names_sorted_rmse = sorted(MODEL_NAMES, key=lambda n: holdout_metrics[n]['RMSE'])
rmses  = [holdout_metrics[n]['RMSE']   for n in names_sorted_rmse]
daccs  = [holdout_metrics[n]['DirAcc'] for n in names_sorted_rmse]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

bar_colors = ['#2ecc71' if i == 0 else '#bdc3c7' for i in range(len(names_sorted_rmse))]
ax1.barh(names_sorted_rmse, rmses, color=bar_colors, edgecolor='none')
ax1.set_xlabel('RMSE (lower is better)')
ax1.set_title('Holdout RMSE', fontweight='bold')
for i, v in enumerate(rmses):
    ax1.text(v * 0.02, i, f'  {v:.1f}', va='center', fontsize=9)

best_da = max(daccs)
bar_colors2 = ['#2ecc71' if v == best_da else '#bdc3c7' for v in daccs]
ax2.barh(names_sorted_rmse, daccs, color=bar_colors2, edgecolor='none')
ax2.axvline(50, color='red', linewidth=1.2, linestyle='--', label='Random (50%)')
ax2.set_xlabel('Directional Accuracy % (higher is better)')
ax2.set_title('Holdout Directional Accuracy', fontweight='bold')
for i, v in enumerate(daccs):
    ax2.text(v * 0.02, i, f'  {v:.1f}%', va='center', fontsize=9)
ax2.legend(fontsize=8)

plt.suptitle('Model Comparison — Holdout Set', fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('05_model_comparison')


# ─── PLOT 6: Feature Importance (Tree Models) ────────────────────────────────
print("\n[PLOT 6] Feature importance (tree models)...")
tree_models = [(n, m) for n, m in models.items()
               if hasattr(m, 'feature_importances_')]
if HAS_LGB:
    tree_models = [t for t in tree_models] + \
                  [('LightGBM', models['LightGBM'])] if 'LightGBM' not in [t[0] for t in tree_models] else tree_models

n_tree = len(tree_models)
fig, axes = plt.subplots(1, n_tree, figsize=(7 * n_tree, 5))
if n_tree == 1:
    axes = [axes]
for ax, (name, mdl) in zip(axes, tree_models):
    imps = mdl.feature_importances_
    order = np.argsort(imps)
    ax.barh([FEATURES[i] for i in order], [imps[i] for i in order], color='#3498db', edgecolor='none')
    ax.set_title(f'{name} Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance')
plt.tight_layout()
save_fig('06_feature_importance')


# ─── PLOT 7: Linear Model Coefficients ───────────────────────────────────────
print("\n[PLOT 7] Linear model coefficients (Ridge & ElasticNet)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ['Ridge', 'ElasticNet']):
    coefs = models[name].named_steps['model'].coef_
    order = np.argsort(np.abs(coefs))
    bar_c = ['#e74c3c' if coefs[i] < 0 else '#2ecc71' for i in order]
    ax.barh([FEATURES[i] for i in order], [coefs[i] for i in order], color=bar_c, edgecolor='none')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'{name} Coefficients (standardised features)', fontweight='bold')
    ax.set_xlabel('Coefficient')
plt.tight_layout()
save_fig('07_linear_coefficients')


# ─── STEP 12: RETRAIN ON FULL DATASET ────────────────────────────────────────
print("\n[STEP 12] Retraining all models on full dataset for forecasting...")
for name, mdl in models.items():
    mdl.fit(X, y)


# ─── STEP 13: 6-MONTH FORWARD FORECAST ───────────────────────────────────────
print(f"\n[STEP 13] {FORECAST_MONTHS}-month forward forecast...")

last_row   = model_df.iloc[-1]
last_date  = last_row['date']
future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(1),
                              periods=FORECAST_MONTHS, freq='ME')

# Assumption: macro features held at last known value (no external macro forecast).
# Only lag1_return and lag1_volume are propagated from predictions.
# This is documented as a known simplification — direction signal, not magnitude.
base_feats  = {f: float(last_row[f]) for f in FEATURES}
prev_return = float(last_row['log_return'])
prev_volume = float(last_row['total_volume'])

forecast_rows = []
for i, fdate in enumerate(future_dates):
    row = base_feats.copy()
    row['lag1_return'] = prev_return
    row['lag1_volume'] = prev_volume

    X_fut    = np.array([[row[f] for f in FEATURES]])
    m_preds  = {name: float(mdl.predict(X_fut)[0]) for name, mdl in models.items()}
    ensemble = float(np.mean(list(m_preds.values())))

    n_pos     = sum(1 for p in m_preds.values() if p > 0)
    direction = 'INFLOW  ↑' if ensemble > 0 else 'OUTFLOW ↓'
    consensus = f'{n_pos}/{len(models)} models → inflow'

    forecast_rows.append({
        'date': fdate, 'ensemble': ensemble,
        'direction': direction, 'consensus': consensus,
        **m_preds
    })

    # Propagate: ensemble inflow/outflow maps to mild return estimate
    # (conservative: 1% × normalised impact; clipped to avoid compounding errors)
    flow_normalised = ensemble / (abs(y).mean() + 1e-9)
    prev_return = float(np.clip(flow_normalised * 0.01, -0.05, 0.05))
    # volume: held constant (no reliable update mechanism)

forecast_df = pd.DataFrame(forecast_rows)

print()
print("  ┌─────────────┬──────────────┬────────────────┬────────────────────────┐")
print("  │ Month       │ Ensemble (PKR)│ Direction      │ Consensus              │")
print("  ├─────────────┼──────────────┼────────────────┼────────────────────────┤")
for _, r in forecast_df.iterrows():
    print(f"  │ {r['date'].strftime('%b %Y'):<11} │ {r['ensemble']:>+12.1f} │ {r['direction']:<14} │ {r['consensus']:<22} │")
print("  └─────────────┴──────────────┴────────────────┴────────────────────────┘")
print("\n  ⚠  Macro features held constant — direction signal only, not magnitude.")


# ─── PLOT 8: 6-Month Forecast ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))
hist12 = model_df.tail(12)
ax.bar(hist12['date'], hist12[TARGET],
       color=np.where(hist12[TARGET] >= 0, '#2ecc71', '#e74c3c'), alpha=0.75, label='Historical')
fcolors = ['#27ae60' if v > 0 else '#c0392b' for v in forecast_df['ensemble']]
ax.bar(forecast_df['date'], forecast_df['ensemble'], color=fcolors, alpha=0.9,
       label='Ensemble forecast', hatch='//')
ax.axhline(0, color='black', linewidth=0.7)
ax.axvline(last_date, color='navy', linewidth=1.5, linestyle='--', label='Last observation')
for (name, color) in zip(MODEL_NAMES, MODEL_COLORS):
    if name in forecast_df.columns:
        ax.plot(forecast_df['date'], forecast_df[name], 'o--', color=color,
                linewidth=1, markersize=4, label=name, alpha=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, ha='right')
ax.set_title(f'{FORECAST_MONTHS}-Month Fund Flow Forecast — Ensemble + Per-Model',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Fund Flow (PKR)')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
save_fig('08_forecast_6month')


# ─── STEP 14: COMPANY SELECTION ──────────────────────────────────────────────
print("\n[STEP 14] Selecting portfolio candidates from KSE-30...")

# Months where actual fund flow was positive
pos_periods = set(model_df.loc[model_df[TARGET] > 0, 'date'].dt.to_period('M'))

co_pos = company_monthly[
    company_monthly['month'].isin(pos_periods)
].dropna(subset=['return'])

# By return rank
avg_ret_pos  = co_pos.groupby('company')['return'].mean().sort_values(ascending=False)
top10_ret    = avg_ret_pos.head(10).reset_index()
top10_ret.columns = ['Company', 'Avg Return']

# By volume rank (fallback)
vol_pos    = co_raw[co_raw['date'].dt.to_period('M').isin(pos_periods)]
avg_vol_pos = vol_pos.groupby('company')['volume'].mean().sort_values(ascending=False)
top10_vol   = avg_vol_pos.head(10).reset_index()
top10_vol.columns = ['Company', 'Avg Volume']

print("\n  Top 10 by return during positive flow months:")
print("  " + top10_ret.to_string(index=False))
print("\n  Top 10 by volume during positive flow months:")
print("  " + top10_vol.to_string(index=False))


# ─── PLOT 9: Company Selection ───────────────────────────────────────────────
next_dir = forecast_df['direction'].iloc[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.barh(top10_ret['Company'][::-1],
         top10_ret['Avg Return'][::-1], color='#27ae60', edgecolor='none')
ax1.set_xlabel('Avg Monthly Return (during +flow months)')
ax1.set_title(f'Top 10 by Return\nNext month signal: {next_dir}', fontweight='bold')

ax2.barh(top10_vol['Company'][::-1],
         top10_vol['Avg Volume'][::-1], color='#2980b9', edgecolor='none')
ax2.set_xlabel('Avg Volume (during +flow months)')
ax2.set_title('Top 10 by Volume\n(liquidity-based fallback)', fontweight='bold')

plt.suptitle('KSE-30 Portfolio Candidates — Positive Flow Months Analysis',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('09_company_selection')


# ─── STEP 15: MARKET EFFICIENCY TESTS ────────────────────────────────────────
print("\n[STEP 15] Market efficiency tests (Hurst Exponent + Variance Ratio)...")
log_ret_series = model_df['log_return'].dropna().values
hurst = hurst_exponent(log_ret_series)
vr    = variance_ratio(log_ret_series)

print(f"    Hurst Exponent : {hurst:.3f}  "
      f"({'persistent' if hurst > 0.55 else 'mean-reverting' if hurst < 0.45 else 'near random walk'})")
print(f"    Variance Ratio : {vr:.3f}  "
      f"({'positive autocorrelation' if vr > 1.1 else 'near random walk'})")
if hurst > 0.5 and vr > 1.0:
    print("    → Momentum signals should carry predictive value for this index.")


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
best_cv   = min(MODEL_NAMES, key=lambda n: np.mean(cv_results[n]['rmse']))
best_hold = min(MODEL_NAMES, key=lambda n: holdout_metrics[n]['RMSE'])
best_da   = max(MODEL_NAMES, key=lambda n: holdout_metrics[n]['DirAcc'])

print("\n" + "═" * 62)
print("  ENHANCED-V5 — SUMMARY")
print("═" * 62)
print(f"  Training window : {model_df['date'].min().date()} → {model_df['date'].max().date()}")
print(f"  Rows used       : {len(model_df)}  (post-2020 drop + NaN cleanup)")
print(f"  Features        : {len(FEATURES)}")
print(f"  Models          : {', '.join(MODEL_NAMES)}")
print(f"\n  Best CV RMSE    : {best_cv}  "
      f"(mean RMSE = {np.mean(cv_results[best_cv]['rmse']):.1f})")
print(f"  Best holdout    : {best_hold}  "
      f"(RMSE = {holdout_metrics[best_hold]['RMSE']:.1f})")
print(f"  Best direction  : {best_da}  "
      f"(DA = {holdout_metrics[best_da]['DirAcc']:.0f}%)")
print(f"\n  Next month pred : {forecast_df.iloc[0]['direction']}")
print(f"  Ensemble value  : {forecast_df.iloc[0]['ensemble']:+.1f} PKR")
print(f"  Consensus       : {forecast_df.iloc[0]['consensus']}")
print(f"\n  Hurst Exponent  : {hurst:.3f}")
print(f"  Variance Ratio  : {vr:.3f}")
print("═" * 62)
print(f"\n  All plots saved → {OUT_DIR}")
print("  ⚠  Use directional signal only — magnitude unreliable at this dataset size.")
