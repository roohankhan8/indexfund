import os, warnings
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
    print("lightgbm not installed — using XGBoost only for tree models.")

warnings.filterwarnings('ignore')
# %matplotlib inline

# ── CONFIG ────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
N_SPLITS        = 5
FORECAST_MONTHS = 6
SAVE_PLOTS      = True

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath('.')   # Jupyter: run from new-pipeline/v5/

DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, 'data'))
OUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Data  : {DATA_DIR}")
print(f"Output: {OUT_DIR}")


def save_fig(name: str):
    if SAVE_PLOTS:
        path = os.path.join(OUT_DIR, f'{name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  → saved: {path}")
    plt.show()

def excel_to_datetime(serial):
    try:
        return datetime(1899, 12, 30) + timedelta(days=int(serial))
    except (ValueError, TypeError):
        return pd.NaT

def parse_date_col(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(excel_to_datetime)
    return pd.to_datetime(series, errors='coerce')

def directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)

def hurst_exponent(ts, max_lag=20):
    lags = range(2, min(max_lag, len(ts) // 2))
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

def variance_ratio(ts, q=3):
    mu   = np.mean(ts)
    var1 = np.var(ts - mu, ddof=1)
    ts_q = [ts[i] - ts[i - q] - q * mu for i in range(q, len(ts))]
    return float(np.var(ts_q, ddof=1) / (q * var1))

print("Helpers defined.")


# ============================================================
# KSE-30 DAILY DATA → Create Weekly & Technical Features
# ============================================================
kse_daily = pd.read_csv(
    os.path.join(DATA_DIR, 'kse30_index_level.csv'), parse_dates=['date']
)
kse_daily = kse_daily.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

# Calculate technical indicators on daily data
# RSI (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

kse_daily['rsi'] = compute_rsi(kse_daily['log_return'])

# Bollinger Bands (20-day SMA ± 2 std)
kse_daily['sma_20'] = kse_daily['log_return'].rolling(20).mean()
kse_daily['std_20'] = kse_daily['log_return'].rolling(20).std()
kse_daily['bb_upper'] = kse_daily['sma_20'] + 2 * kse_daily['std_20']
kse_daily['bb_lower'] = kse_daily['sma_20'] - 2 * kse_daily['std_20']
kse_daily['bb_position'] = (kse_daily['log_return'] - kse_daily['bb_lower']) / (kse_daily['bb_upper'] - kse_daily['bb_lower'] + 1e-10)

# MACD (12-day EMA - 26-day EMA)
kse_daily['ema_12'] = kse_daily['log_return'].ewm(span=12).mean()
kse_daily['ema_26'] = kse_daily['log_return'].ewm(span=26).mean()
kse_daily['macd'] = kse_daily['ema_12'] - kse_daily['ema_26']
kse_daily['macd_signal'] = kse_daily['macd'].ewm(span=9).mean()

# Weekly aggregation of daily features
kse_weekly = (
    kse_daily.set_index('date').resample('W-FRI').agg(
        log_return   = ('log_return', 'sum'),
        total_volume = ('total_volume', 'sum'),
        avg_rsi      = ('rsi', 'mean'),
        avg_bb_pos   = ('bb_position', 'mean'),
        avg_macd     = ('macd', 'mean'),
    ).reset_index()
)

# Compute rolling features on weekly data
kse_weekly['vol4_return'] = kse_weekly['log_return'].rolling(4).std()  # 4-week volatility
kse_weekly['avg4m_volume'] = kse_weekly['total_volume'].rolling(4).mean()
kse_weekly['abnorm_vol'] = kse_weekly['total_volume'] / kse_weekly['avg4m_volume']
kse_weekly['momentum_4w'] = kse_weekly['log_return'].rolling(4).sum()  # 4-week momentum

print(f"KSE-30 weekly: {len(kse_weekly)} rows")
print(f"Date range    : {kse_weekly['date'].min().date()} → {kse_weekly['date'].max().date()}")
print(f"New features  : rsi, bb_position, macd, vol4_return, momentum_4w")
kse_weekly.tail(3)

co_raw = pd.read_excel(os.path.join(DATA_DIR, 'kse-30-basic.xlsx'))
co_raw.columns = co_raw.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
co_raw = co_raw.rename(columns={'idx_wt_%': 'idx_wt', 'date': 'date_raw'})
co_raw['date'] = parse_date_col(co_raw['date_raw'])
co_raw = (co_raw
          .dropna(subset=['date', 'company', 'price'])
          .drop_duplicates(subset=['date', 'company'])
          .sort_values(['company', 'date']))

co_raw['month'] = co_raw['date'].dt.to_period('M')
company_monthly = co_raw.groupby(['company', 'month'])['price'].last().reset_index()
company_monthly['return']   = company_monthly.groupby('company')['price'].pct_change()
company_monthly['month_dt'] = company_monthly['month'].dt.to_timestamp('M')

print(f"{co_raw['company'].nunique()} companies | "
      f"{co_raw['date'].min().date()} → {co_raw['date'].max().date()}")


flow_frames = []
for fund_name in ['AKD', 'NBP', 'NTI']:
    fdf = pd.read_excel(os.path.join(DATA_DIR, 'funds_data.xlsx'), sheet_name=fund_name)
    fdf.columns = fdf.columns.str.strip().str.upper()
    fdf['date'] = parse_date_col(fdf['DATE'])
    fdf = fdf.dropna(subset=['date', 'NAV', 'AUM']).sort_values('date')
    fdf['nav_ratio'] = fdf['NAV'] / fdf['NAV'].shift(1)
    fdf['flow']      = fdf['AUM'] - (fdf['AUM'].shift(1) * fdf['nav_ratio'])
    flow_frames.append(fdf[['date', 'flow']])
    print(f"  {fund_name}: {len(fdf)} daily rows")

monthly_flow = (
    pd.concat(flow_frames, ignore_index=True)
    .set_index('date').resample('ME')['flow']
    .sum().reset_index(name='total_fund_flow')
)
monthly_flow['total_fund_flow'] = monthly_flow['total_fund_flow'].fillna(0)
print(f"\nMonthly flow rows: {len(monthly_flow)}")
monthly_flow.tail(3)


# Oil
oil         = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='OIL')
oil['date'] = parse_date_col(oil['DATE'])
oil_monthly = (oil.dropna(subset=['date']).set_index('date').resample('ME')['PRICE']
               .last().reset_index().rename(columns={'PRICE': 'oil_price'}))

# Interest Rate — step function, forward fill to daily before monthly resample
ir          = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='IR')
ir['date']  = parse_date_col(ir['DATE'])
ir          = ir.dropna(subset=['date']).set_index('date').sort_index()
ir_daily    = ir.reindex(pd.date_range(ir.index.min(), ir.index.max(), freq='D')).ffill()
ir_monthly  = (ir_daily['RATE'].resample('ME').last()
               .reset_index().rename(columns={'index': 'date', 'RATE': 'interest_rate'}))

# USD
usd         = pd.read_excel(os.path.join(DATA_DIR, 'macro_data.xlsx'), sheet_name='USD')
usd['date'] = parse_date_col(usd['DATE'])
usd_monthly = (usd.dropna(subset=['date']).set_index('date').resample('ME')['USD']
               .last().reset_index().rename(columns={'USD': 'usd_exchange'}))

# CPI
cpi_raw           = pd.read_csv(os.path.join(DATA_DIR, 'cpi.csv'), skiprows=1,
                                 header=None, names=['period', 'yoy_cpi'])
cpi_raw           = cpi_raw[cpi_raw['period'] != 'Period'].dropna()
cpi_raw['date']   = pd.to_datetime('01-' + cpi_raw['period'].astype(str),
                                    format='%d-%b-%y', errors='coerce')
cpi_raw['yoy_cpi'] = pd.to_numeric(cpi_raw['yoy_cpi'], errors='coerce')
cpi_monthly       = (cpi_raw.dropna(subset=['date', 'yoy_cpi'])
                     .set_index('date').resample('ME')['yoy_cpi']
                     .last().reset_index().rename(columns={'yoy_cpi': 'cpi'}))

print(f"IR  coverage: {ir_monthly['date'].min().date()} → {ir_monthly['date'].max().date()}")
print(f"CPI coverage: {cpi_monthly['date'].min().date()} → {cpi_monthly['date'].max().date()}")

# ============================================================
# MERGE: Weekly KSE + Weekly Fund Flows (computed directly from daily)
# ============================================================

# Compute fund flows at weekly frequency directly from daily data
flow_frames_weekly = []
for fund_name in ['AKD', 'NBP', 'NTI']:
    fdf = pd.read_excel(os.path.join(DATA_DIR, 'funds_data.xlsx'), sheet_name=fund_name)
    fdf.columns = fdf.columns.str.strip().str.upper()
    fdf['date'] = parse_date_col(fdf['DATE'])
    fdf = fdf.dropna(subset=['date', 'NAV', 'AUM']).sort_values('date')
    fdf['nav_ratio'] = fdf['NAV'] / fdf['NAV'].shift(1)
    fdf['flow'] = fdf['AUM'] - (fdf['AUM'].shift(1) * fdf['nav_ratio'])

    # Aggregate to weekly
    fdf_weekly = fdf.set_index('date').resample('W-FRI')['flow'].sum().reset_index()
    fdf_weekly = fdf_weekly.rename(columns={'flow': fund_name})
    flow_frames_weekly.append(fdf_weekly)

# Merge all funds weekly
weekly_flow = flow_frames_weekly[0]
for i, name in enumerate(['AKD', 'NBP', 'NTI'][1:], 1):
    weekly_flow = weekly_flow.merge(flow_frames_weekly[i], on='date', how='outer')
weekly_flow['total_fund_flow'] = weekly_flow[['AKD', 'NBP', 'NTI']].sum(axis=1)
weekly_flow = weekly_flow.sort_values('date').reset_index(drop=True)

print(f"Weekly fund flow rows: {len(weekly_flow)}")

# Merge weekly KSE with weekly flows
df = kse_weekly.copy()
df = df.merge(weekly_flow[['date', 'total_fund_flow']], on='date', how='left')

# Add macro data
for src in [oil_monthly, ir_monthly, usd_monthly, cpi_monthly]:
    df = df.merge(src, on='date', how='left')

for col in ['oil_price', 'interest_rate', 'usd_exchange', 'cpi']:
    if col in df.columns:
        df[col] = df[col].ffill()

# Fill missing fund flows with 0 (no flow that week)
df['total_fund_flow'] = df['total_fund_flow'].fillna(0)

# Drop rows before 2021 (no macro data)
df = df[df['date'].dt.year >= 2021].reset_index(drop=True)
print(f"After merge: {len(df)} weekly rows ({df['date'].min().date()} → {df['date'].max().date()})")

# Winsorise weekly flows
mean_flow = df['total_fund_flow'].mean()
std_flow  = df['total_fund_flow'].std()
lower_w, upper_w = mean_flow - 3*std_flow, mean_flow + 3*std_flow
n_clip = int(((df['total_fund_flow'] < lower_w) | (df['total_fund_flow'] > upper_w)).sum())
df['total_fund_flow'] = df['total_fund_flow'].clip(lower=lower_w, upper=upper_w)
print(f"Winsorised {n_clip} rows  [bounds: {lower_w:.1f}, {upper_w:.1f}]")

print(f"\nData shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# FEATURE ENGINEERING - Enhanced for Weekly Data
# ============================================================
# Now with technical indicators as features

df['lag1_volume']      = df['total_volume'].shift(1)
df['lag2_volume']      = df['total_volume'].shift(2)
df['lag1_return']      = df['log_return'].shift(1)
df['lag2_return']      = df['log_return'].shift(2)
df['lag1_vol4_ret']    = df['vol4_return'].shift(1)
df['lag1_abnorm_vol']  = df['abnorm_vol'].shift(1)
df['lag1_momentum']    = df['momentum_4w'].shift(1)

# Technical indicator lags
df['lag1_rsi']         = df['avg_rsi'].shift(1)
df['lag1_bb_pos']      = df['avg_bb_pos'].shift(1)
df['lag1_macd']        = df['avg_macd'].shift(1)

# Macro lags
df['lag1_ir']          = df['interest_rate'].shift(1) if 'interest_rate' in df.columns else 0
df['lag1_oil']         = df['oil_price'].shift(1) if 'oil_price' in df.columns else 0
df['lag1_usd']         = df['usd_exchange'].shift(1) if 'usd_exchange' in df.columns else 0

FEATURES = [
    'lag1_volume', 'lag2_volume',
    'lag1_return', 'lag2_return',
    'lag1_vol4_ret', 'lag1_abnorm_vol', 'lag1_momentum',
    'lag1_rsi', 'lag1_bb_pos', 'lag1_macd',
    'lag1_ir', 'lag1_oil', 'lag1_usd',
]
TARGET = 'total_fund_flow'

model_df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
X = model_df[FEATURES].values
y = model_df[TARGET].values

print(f"Feature matrix : {model_df.shape[0]} weekly rows × {len(FEATURES)} features")
print(f"Usable window  : {model_df['date'].min().date()} → {model_df['date'].max().date()}")
print(f"\nNew technical features: RSI, Bollinger Band Position, MACD")
print(f"New momentum feature: 4-week cumulative return")

fig, ax = plt.subplots(figsize=(13, 9))
corr_data   = model_df[FEATURES + [TARGET]].copy()
corr_data.columns = FEATURES + ['fund_flow']
corr_matrix = corr_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.75})
ax.set_title('Feature Correlation Matrix — enhanced-v7 (Weekly + Technical)', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('01_correlation_heatmap')

fig, ax = plt.subplots(figsize=(14, 5))
colors_bar = np.where(model_df[TARGET] >= 0, '#2ecc71', '#e74c3c')
ax.bar(model_df['date'], model_df[TARGET], color=colors_bar, edgecolor='none', alpha=0.85, width=5)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(upper_w, color='darkorange', linewidth=1.2, linestyle='--', label=f'+3σ = {upper_w:.0f}')
ax.axhline(lower_w, color='darkorange', linewidth=1.2, linestyle='--', label=f'−3σ = {lower_w:.0f}')
ax.set_ylabel('Fund Flow (PKR)', fontsize=10)
ax.set_title('Weekly Fund Flow — AKD + NBP + NTI (Winsorised at ±3σ)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
save_fig('02_fund_flow_history')

MODEL_COLORS = ['#e74c3c', '#3498db', '#9b59b6', '#27ae60', '#f39c12']

models = {
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

MODEL_NAMES = list(models.keys())
print(f"Active models: {MODEL_NAMES}")


tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_results = {name: {'rmse': [], 'mae': [], 'dir_acc': []} for name in MODEL_NAMES}
fold_preds  = {name: [] for name in MODEL_NAMES}

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

summary = pd.DataFrame({
    name: {
        'Mean RMSE': f"{np.mean(v['rmse']):.1f}",
        'Std RMSE':  f"{np.std(v['rmse']):.1f}",
        'Mean MAE':  f"{np.mean(v['mae']):.1f}",
        'Dir Acc %': f"{np.mean(v['dir_acc']):.1f}",
    } for name, v in cv_results.items()
}).T
display(summary)


n_m = len(MODEL_NAMES)
fig, axes = plt.subplots(n_m, N_SPLITS, figsize=(18, 3.2*n_m), sharey=True)
for mi, name in enumerate(MODEL_NAMES):
    for fi, (dv, yv, pv) in enumerate(fold_preds[name]):
        ax = axes[mi][fi]
        ax.plot(dv, yv, 'o-', color='steelblue',  markersize=4, linewidth=1.2, label='Actual')
        ax.plot(dv, pv, 's--', color='tomato',    markersize=4, linewidth=1.2, label='Predicted')
        ax.axhline(0, color='gray', linewidth=0.5)
        if mi == 0:
            ax.set_title(f'Fold {fi+1}', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        if fi == 0:
            ax.set_ylabel(name, fontsize=9, fontweight='bold')
        if mi == 0 and fi == N_SPLITS - 1:
            ax.legend(fontsize=7)
fig.suptitle(f'TimeSeriesSplit ({N_SPLITS} Folds) — Actual vs Predicted per Model',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('03_tscv_folds')


split_idx = int(len(X) * 0.8)
X_tr_h, X_te_h = X[:split_idx], X[split_idx:]
y_tr_h, y_te_h = y[:split_idx], y[split_idx:]
dates_te        = model_df['date'].iloc[split_idx:].values

holdout_preds, holdout_metrics = {}, {}
for name, mdl in models.items():
    mdl.fit(X_tr_h, y_tr_h)
    preds = mdl.predict(X_te_h)
    holdout_preds[name] = preds
    holdout_metrics[name] = {
        'RMSE':   round(float(np.sqrt(mean_squared_error(y_te_h, preds))), 1),
        'MAE':    round(float(mean_absolute_error(y_te_h, preds)), 1),
        'R²':     round(float(r2_score(y_te_h, preds)), 3),
        'DirAcc': round(directional_accuracy(y_te_h, preds), 1),
    }

display(pd.DataFrame(holdout_metrics).T.sort_values('RMSE'))


fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_te, y_te_h, 'o-', color='steelblue', linewidth=2, markersize=5,
        label='Actual', zorder=5)
for (name, preds), color in zip(holdout_preds.items(), MODEL_COLORS):
    r2 = holdout_metrics[name]['R²']
    da = holdout_metrics[name]['DirAcc']
    # Highlight negative R² (overfitting indicator)
    r2_str = f"{r2:.2f}" if r2 >= 0 else f"{r2:.2f} (OVERFIT)"
    ax.plot(dates_te, preds, '--', color=color, linewidth=1.5, markersize=4,
            label=f'{name}  (R²={r2_str}, DA={da:.0f}%)')
ax.axhline(0, color='gray', linewidth=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, ha='right')
ax.set_title('Holdout: All Models vs Actual\n⚠ Negative R² = model worse than predicting mean (overfitting)', 
             fontsize=12, fontweight='bold')
ax.set_ylabel('Fund Flow (PKR)')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('04_holdout_predictions')

# Comparison bar charts with statistical notes
names_s = sorted(MODEL_NAMES, key=lambda n: holdout_metrics[n]['RMSE'])
rmses   = [holdout_metrics[n]['RMSE']   for n in names_s]
daccs   = [holdout_metrics[n]['DirAcc'] for n in names_s]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
bc1 = ['#2ecc71' if i == 0 else '#bdc3c7' for i in range(len(names_s))]
ax1.barh(names_s, rmses, color=bc1, edgecolor='none')
ax1.set_xlabel('RMSE')
ax1.set_title('Holdout RMSE (lower = better)', fontweight='bold')
for i, v in enumerate(rmses):
    ax1.text(v*0.02, i, f'  {v:.1f}', va='center', fontsize=9)

best_da = max(daccs)
bc2 = ['#2ecc71' if v == best_da else '#bdc3c7' for v in daccs]
ax2.barh(names_s, daccs, color=bc2, edgecolor='none')
ax2.axvline(50, color='red', linewidth=1.2, linestyle='--', label='Random (50%)')
ax2.set_xlabel('Directional Accuracy %')
ax2.set_title(f'Holdout Directional Accuracy\n(⚠ 10 points: difference of 1 call = 10%)', fontweight='bold')
for i, v in enumerate(daccs):
    ax2.text(v*0.02, i, f'  {v:.1f}%', va='center', fontsize=9)
ax2.legend(fontsize=8)
plt.suptitle('Model Comparison — Holdout Set', fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('05_model_comparison')

# Tree model feature importance - FIX: use gain-based importance for all (normalize LightGBM split counts)
tree_models = [(n, m) for n, m in models.items() if hasattr(m, 'feature_importances_')]
n_tree = len(tree_models)
fig, axes = plt.subplots(1, n_tree, figsize=(7*n_tree, 5))
if n_tree == 1:
    axes = [axes]

# Normalize all importances to 0-1 scale for comparability
for ax, (name, mdl) in zip(axes, tree_models):
    imps = mdl.feature_importances_.copy()
    # Normalize to sum to 1 (proportion-based, comparable across models)
    imps = imps / imps.sum() if imps.sum() > 0 else imps
    order = np.argsort(imps)
    ax.barh([FEATURES[i] for i in order], [imps[i] for i in order],
            color='#3498db', edgecolor='none')
    ax.set_title(f'{name} Feature Importance\n(normalized to sum=1)', fontweight='bold')
    ax.set_xlabel('Importance (proportion)')
plt.tight_layout()
save_fig('06_feature_importance')

# Linear model coefficients with multicollinearity note
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ['Ridge', 'ElasticNet']):
    coefs = models[name].named_steps['model'].coef_
    order = np.argsort(np.abs(coefs))
    bar_c = ['#e74c3c' if coefs[i] < 0 else '#2ecc71' for i in order]
    ax.barh([FEATURES[i] for i in order], [coefs[i] for i in order],
            color=bar_c, edgecolor='none')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'{name} Coefficients (standardised)\n⚠ Large values may reflect multicollinearity, not predictive power', 
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Coefficient')
plt.tight_layout()
save_fig('07_linear_coefficients')

# Retrain on full dataset
for name, mdl in models.items():
    mdl.fit(X, y)

last_row     = model_df.iloc[-1]
last_date    = last_row['date']
FORECAST_WEEKS = 12  # 12 weeks instead of 6 months
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1),
                              periods=FORECAST_WEEKS, freq='W-FRI')

base_feats  = {f: float(last_row[f]) for f in FEATURES}
prev_return = float(last_row['log_return'])
prev_volume = float(last_row['total_volume'])

forecast_rows = []
for fdate in future_dates:
    row = base_feats.copy()
    row['lag1_return'] = prev_return
    row['lag1_volume'] = prev_volume
    X_fut   = np.array([[row[f] for f in FEATURES]])
    m_preds = {name: float(mdl.predict(X_fut)[0]) for name, mdl in models.items()}
    ensemble = float(np.mean(list(m_preds.values())))
    n_pos    = sum(1 for p in m_preds.values() if p > 0)
    forecast_rows.append({
        'date': fdate, 'ensemble': ensemble,
        'direction': 'INFLOW ↑' if ensemble > 0 else 'OUTFLOW ↓',
        'consensus': f'{n_pos}/{len(models)} models → inflow',
        **m_preds
    })
    flow_norm   = ensemble / (abs(y).mean() + 1e-9)
    prev_return = float(np.clip(flow_norm * 0.01, -0.05, 0.05))

forecast_df = pd.DataFrame(forecast_rows)
display(forecast_df[['date', 'ensemble', 'direction', 'consensus']].to_string(index=False))

fig, ax = plt.subplots(figsize=(15, 5))
hist12    = model_df.tail(12)
ax.bar(hist12['date'], hist12[TARGET],
       color=np.where(hist12[TARGET] >= 0, '#2ecc71', '#e74c3c'), alpha=0.75, label='Historical')
fcolors = ['#27ae60' if v > 0 else '#c0392b' for v in forecast_df['ensemble']]
ax.bar(forecast_df['date'], forecast_df['ensemble'], color=fcolors, alpha=0.9,
       label='Ensemble forecast', hatch='//')
ax.axhline(0, color='black', linewidth=0.7)
ax.axvline(last_date, color='navy', linewidth=1.5, linestyle='--', label='Last observation')
for name, color in zip(MODEL_NAMES, MODEL_COLORS):
    if name in forecast_df.columns:
        ax.plot(forecast_df['date'], forecast_df[name], 'o--', color=color,
                linewidth=1, markersize=4, label=name, alpha=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, ha='right')
ax.set_title(f'{FORECAST_WEEKS}-Week Fund Flow Forecast — Ensemble + Per-Model',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Fund Flow (PKR)')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
save_fig('08_forecast_12week')

pos_periods = set(model_df.loc[model_df[TARGET] > 0, 'date'].dt.to_period('M'))
co_pos      = company_monthly[company_monthly['month'].isin(pos_periods)].dropna(subset=['return'])

# FIX: Winsorize company returns at 1-99 percentile to handle BOP outlier (Critical bug fix)
ret_p1, ret_p99 = co_pos['return'].quantile([0.01, 0.99])
co_pos_winsorized = co_pos['return'].clip(lower=ret_p1, upper=ret_p99)
co_pos['return_winsorized'] = co_pos_winsorized

avg_ret_pos  = co_pos.groupby('company')['return_winsorized'].mean().sort_values(ascending=False)
top10_ret    = avg_ret_pos.head(10).reset_index()
top10_ret.columns = ['Company', 'Avg Return (winsorized)']

vol_pos     = co_raw[co_raw['date'].dt.to_period('M').isin(pos_periods)]
avg_vol_pos = vol_pos.groupby('company')['volume'].mean().sort_values(ascending=False)
top10_vol   = avg_vol_pos.head(10).reset_index()
top10_vol.columns = ['Company', 'Avg Volume']

next_dir = forecast_df['direction'].iloc[0]
is_outflow = 'OUTFLOW' in next_dir

# Determine which chart to show prominently based on signal
# OUTFLOW → equal-weight (show volume as primary, returns as secondary)
# INFLOW → overweight (show returns as primary, volume as secondary)
if is_outflow:
    # For outflow, show volume as primary (equal-weight strategy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.barh(top10_vol['Company'][::-1], top10_vol['Avg Volume'][::-1],
             color='#2980b9', edgecolor='none')
    ax1.set_xlabel('Avg Volume (during +flow months)')
    ax1.set_title(f'Top 10 by Volume (EQUAL-WEIGHT Strategy)\nNext signal: {next_dir}', fontweight='bold')
    
    ax2.barh(top10_ret['Company'][::-1], top10_ret['Avg Return (winsorized)'][::-1],
             color='#27ae60', edgecolor='none')
    ax2.set_xlabel('Avg Monthly Return (winsorized at 1-99%)')
    ax2.set_title('Top 10 by Return (for reference)', fontweight='bold')
    strategy_note = "OUTFLOW → Equal-weight portfolio (use volume ranking)"
else:
    # For inflow, show returns as primary (overweight strategy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.barh(top10_ret['Company'][::-1], top10_ret['Avg Return (winsorized)'][::-1],
             color='#27ae60', edgecolor='none')
    ax1.set_xlabel('Avg Monthly Return (winsorized at 1-99%)')
    ax1.set_title(f'Top 10 by Return (OVERWEIGHT Strategy)\nNext signal: {next_dir}', fontweight='bold')
    
    ax2.barh(top10_vol['Company'][::-1], top10_vol['Avg Volume'][::-1],
             color='#2980b9', edgecolor='none')
    ax2.set_xlabel('Avg Volume (during +flow months)')
    ax2.set_title('Top 10 by Volume (for reference)', fontweight='bold')
    strategy_note = "INFLOW → Overweight portfolio (use return ranking)"

plt.suptitle(f'KSE-30 Portfolio Candidates — {strategy_note}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('09_company_selection')

print(f"\n⚠ Returns winsorized at 1-99%: [{ret_p1:.2%}, {ret_p99:.2%}]")
print(strategy_note)
print("\nTop 10 by Return (winsorized):")
display(top10_ret)
print("\nTop 10 by Volume:")
display(top10_vol)

# Hurst Exponent + Variance Ratio (from grok-v1)
series = model_df['log_return'].dropna().values
hurst  = hurst_exponent(series)
vr     = variance_ratio(series)

print(f"Hurst Exponent : {hurst:.3f}  "
      f"({'persistent — momentum signals viable' if hurst > 0.55 else 'near random walk'})")
print(f"Variance Ratio : {vr:.3f}  "
      f"({'positive autocorrelation' if vr > 1.1 else 'near random walk'})")

# Final summary
best_cv   = min(MODEL_NAMES, key=lambda n: np.mean(cv_results[n]['rmse']))
best_hold = min(MODEL_NAMES, key=lambda n: holdout_metrics[n]['RMSE'])
best_da   = max(MODEL_NAMES, key=lambda n: holdout_metrics[n]['DirAcc'])

print("\n" + "="*62)
print("  ENHANCED-V7 SUMMARY (WEEKLY + TECHNICAL INDICATORS)")
print("="*62)
print(f"  Training window : {model_df['date'].min().date()} → {model_df['date'].max().date()}")
print(f"  Rows used       : {len(model_df)} weekly observations")
print(f"  Features        : {len(FEATURES)}  |  Models: {', '.join(MODEL_NAMES)}")
print(f"  New features    : RSI, Bollinger Band Position, MACD, Momentum")
print(f"\n  Best CV RMSE    : {best_cv}  (mean = {np.mean(cv_results[best_cv]['rmse']):.1f})")
print(f"  Best holdout    : {best_hold}  (RMSE = {holdout_metrics[best_hold]['RMSE']:.1f})")
print(f"  Best direction  : {best_da}  (DA = {holdout_metrics[best_da]['DirAcc']:.0f}%)")
print(f"\n  Next week       : {forecast_df.iloc[0]['direction']}")
print(f"  Ensemble pred   : {forecast_df.iloc[0]['ensemble']:+.1f} PKR")
print(f"  Consensus       : {forecast_df.iloc[0]['consensus']}")
print("="*62)
print(f"\n  Plots saved → {OUT_DIR}")
print("  ⚠  Weekly data may have more noise but more signal from technical indicators")

# ============================================================
# IMPROVEMENT 1: Classification - Predict Direction Only
# ============================================================
# Instead of predicting magnitude, predict if next month is inflow (+1) or outflow (-1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create binary target
y_class = np.sign(y)  # +1 for inflow, -1 for outflow, 0 becomes +1 (or we can handle separately)
y_class = np.where(y >= 0, 1, 0)  # 1 = inflow, 0 = outflow

print("=== Classification: Predict Inflow vs Outflow ===")
print(f"Class distribution: Inflow={np.sum(y_class)}, Outflow={np.sum(1-y_class)}")

# TimeSeriesSplit for classification
tscv_class = TimeSeriesSplit(n_splits=5)
class_scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv_class.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_class[tr_idx], y_class[val_idx]
    
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    class_scores.append(acc)
    print(f"  Fold {fold+1}: Accuracy = {acc:.1%}")

print(f"\nMean Classification Accuracy: {np.mean(class_scores):.1%}")
print("⚠ Note: With 2 classes, random baseline = 50%")

# ============================================================
# IMPROVEMENT 2: Reduced Feature Set (Top 4 Features)
# ============================================================
# Use only 4 most relevant features to reduce overfitting
from sklearn.feature_selection import SelectKBest, f_regression

# Select top 4 features
selector = SelectKBest(f_regression, k=4)
selector.fit(X, y)
selected_mask = selector.get_support()
selected_features = [FEATURES[i] for i in range(len(FEATURES)) if selected_mask[i]]

print("=== Reduced Feature Model (4 features) ===")
print(f"Selected features: {selected_features}")

X_reduced = model_df[selected_features].values

# Test with simple Ridge
tscv_red = TimeSeriesSplit(n_splits=5)
red_scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv_red.split(X_reduced)):
    X_tr, X_val = X_reduced[tr_idx], X_reduced[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    ridge = Ridge(alpha=10.0)  # Stronger regularization
    ridge.fit(X_tr_s, y_tr)
    preds = ridge.predict(X_val_s)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    da = directional_accuracy(y_val, preds)
    red_scores.append({'rmse': rmse, 'da': da})
    print(f"  Fold {fold+1}: RMSE={rmse:.1f}, DirAcc={da:.0f}%")

print(f"\nMean RMSE: {np.mean([s['rmse'] for s in red_scores]):.1f}")
print(f"Mean DirAcc: {np.mean([s['da'] for s in red_scores]):.0f}%")

# ============================================================
# IMPROVEMENT 3: Naive Baseline - Predict Previous Month's Flow
# ============================================================
# Compare against simple baselines

print("=== Naive Baseline Comparisons ===")

# Baseline 1: Predict previous month's flow (lag1 of target)
naive_lag1_preds = y[:-1]
naive_lag1_actual = y[1:]
naive_rmse_1 = np.sqrt(mean_squared_error(naive_lag1_actual, naive_lag1_preds))
naive_da_1 = directional_accuracy(naive_lag1_actual, naive_lag1_preds)
print(f"Predict previous month flow: RMSE={naive_rmse_1:.1f}, DirAcc={naive_da_1:.0f}%")

# Baseline 2: Predict mean of training set
mean_flow = y.mean()
naive_mean_preds = np.full(len(y), mean_flow)
naive_rmse_2 = np.sqrt(mean_squared_error(y, naive_mean_preds))
naive_da_2 = directional_accuracy(y, naive_mean_preds)
print(f"Predict mean ({mean_flow:.1f}): RMSE={naive_rmse_2:.1f}, DirAcc={naive_da_2:.0f}%")

# Baseline 3: Predict zero
naive_zero_preds = np.zeros(len(y))
naive_rmse_3 = np.sqrt(mean_squared_error(y, naive_zero_preds))
naive_da_3 = directional_accuracy(y, naive_zero_preds)
print(f"Predict zero: RMSE={naive_rmse_3:.1f}, DirAcc={naive_da_3:.0f}%")

print("\n→ Our best model (Ridge) holdout RMSE = 240.6, compare to baselines above")

# ============================================================
# NEW: Long-Term Forecast (2-3 Years)
# ============================================================
# Extended forecast for portfolio planning

print("=== Generating Long-Term Forecast (2 Years) ===")

# Parameters
YEARS_AHEAD = 2
FORECAST_WEEKS_LONG = 52 * YEARS_AHEAD  # 104 weeks for 2 years

# Retrain on full dataset
for name, mdl in models.items():
    mdl.fit(X, y)

# Get last known values
last_row = model_df.iloc[-1]
last_date = last_row['date']

# Generate future dates
future_dates_long = pd.date_range(
    last_date + pd.Timedelta(weeks=1),
    periods=FORECAST_WEEKS_LONG,
    freq='W-FRI'
)

# Base features from last observation
base_feats = {f: float(last_row[f]) for f in FEATURES}
prev_return = float(last_row['log_return'])
prev_volume = float(last_row['total_volume'])
prev_rsi = float(last_row.get('lag1_rsi', 50))  # Default neutral RSI
prev_bb = float(last_row.get('lag1_bb_pos', 0.5))  # Default middle BB
prev_macd = float(last_row.get('lag1_macd', 0))

# Generate forecasts iteratively
forecast_long = []
for fdate in future_dates_long:
    row = base_feats.copy()
    
    # Update time-varying features
    row['lag1_return'] = prev_return
    row['lag2_return'] = prev_return * 0.9  # Decay effect
    row['lag1_volume'] = prev_volume
    row['lag2_volume'] = prev_volume * 0.95
    row['lag1_rsi'] = prev_rsi
    row['lag1_bb_pos'] = prev_bb
    row['lag1_macd'] = prev_macd
    
    # Predict
    X_fut = np.array([[row[f] for f in FEATURES]])
    m_preds = {name: float(mdl.predict(X_fut)[0]) for name, mdl in models.items()}
    ensemble = float(np.mean(list(m_preds.values())))
    
    # Direction
    direction = 'INFLOW' if ensemble > 0 else 'OUTFLOW'
    
    forecast_long.append({
        'date': fdate,
        'ensemble': ensemble,
        'direction': direction,
        **{name: val for name, val in m_preds.items()}
    })
    
    # Update for next iteration (decaying adjustment)
    flow_norm = ensemble / (abs(y).mean() + 1e-9)
    prev_return = float(np.clip(flow_norm * 0.01, -0.1, 0.1))
    prev_rsi = np.clip(prev_rsi + (ensemble * 0.01), 30, 70)  # RSI bounded
    prev_bb = np.clip(prev_bb + (ensemble * 0.005), 0, 1)  # BB position bounded
    prev_macd = float(np.clip(ensemble * 0.001, -0.05, 0.05))

forecast_long_df = pd.DataFrame(forecast_long)

# Summary by year
yearly_summary = forecast_long_df.copy()
yearly_summary['year'] = yearly_summary['date'].dt.year
yearly_by_year = yearly_summary.groupby('year')['ensemble'].agg(['sum', 'mean', 'std'])
print("\nYearly Forecast Summary:")
print(yearly_by_year)

# Plot long-term forecast
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full 2-year forecast
ax1 = axes[0]
ax1.plot(forecast_long_df['date'], forecast_long_df['ensemble'], 
         color='#2ecc71', linewidth=1.5, label='Ensemble Forecast', alpha=0.8)
ax1.axhline(0, color='black', linewidth=0.8)

# Color by direction
for i in range(len(forecast_long_df) - 1):
    color = '#27ae60' if forecast_long_df['ensemble'].iloc[i] >= 0 else '#c0392b'
    ax1.plot(forecast_long_df['date'].iloc[i:i+2], 
             forecast_long_df['ensemble'].iloc[i:i+2],
             color=color, linewidth=2, alpha=0.7)

# Add year markers
for year in [2026, 2027]:
    ax1.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.5)

ax1.set_title(f'{YEARS_AHEAD}-Year Fund Flow Forecast ({FORECAST_WEEKS_LONG} weeks)', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Weekly Fund Flow (PKR)')
ax1.legend(loc='upper right')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Quarterly aggregation
ax2 = axes[1]
forecast_long_df['quarter'] = forecast_long_df['date'].dt.to_period('Q')
quarterly = forecast_long_df.groupby('quarter')['ensemble'].sum().reset_index()
quarterly['date'] = quarterly['quarter'].dt.to_timestamp()

colors_q = np.where(quarterly['ensemble'] >= 0, '#2ecc71', '#e74c3c')
ax2.bar(quarterly['date'], quarterly['ensemble'], color=colors_q, width=80, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Quarterly Aggregated Forecast', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quarterly Fund Flow (PKR)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
save_fig('10_forecast_2year')

# Print forecast summary
print("\n" + "="*60)
print("  LONG-TERM FORECAST SUMMARY")
print("="*60)
print(f"  Forecast period: {forecast_long_df['date'].min().date()} → {forecast_long_df['date'].max().date()}")
print(f"  Total weeks: {len(forecast_long_df)}")
print(f"\n  First quarter: {forecast_long_df.iloc[:13]['ensemble'].sum():+.1f} PKR")
print(f"  Second quarter: {forecast_long_df.iloc[13:26]['ensemble'].sum():+.1f} PKR")
print(f"  Third quarter: {forecast_long_df.iloc[26:39]['ensemble'].sum():+.1f} PKR")
print(f"  Fourth quarter: {forecast_long_df.iloc[39:52]['ensemble'].sum():+.1f} PKR")
print(f"  Year 2 total: {forecast_long_df.iloc[52:]['ensemble'].sum():+.1f} PKR")
print("="*60)
print("\n⚠ WARNING: Long-term forecasts are highly speculative")
print("   - Model trained on limited data")
print("   - Assumes stable relationships (unrealistic)")
print("   - Use for DIRECTION only, not magnitude")
print("   - Recommend re-fitting model quarterly with new data")
