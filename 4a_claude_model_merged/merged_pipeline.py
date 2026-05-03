"""
COMPREHENSIVE MERGED PIPELINE — All-in-One Analysis
=====================================================
Orchestrates the complete FYP research workflow in a single modular Python script.

Sections:
  0. Configuration & Setup
  1. Preprocessing & Data Cleaning (from nb0)
  2. Exploratory Data Analysis (from nb1)
  3. Fund Flow Prediction (from nb2, nb7)
  4. GARCH Volatility Modelling (from nb3)
  5. Portfolio Optimisation (from nb4)
  6. Rebalancing Prediction (from nb4b)
  7. Market Efficiency Analysis (from nb5)
  8. Results Summary (from nb6)

Input Data:  input_data/
Output Data: output_data/

Run: python merged_pipeline.py
Requirements: pandas, numpy, openpyxl, scipy, scikit-learn, matplotlib, seaborn
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import lstsq as sp_lstsq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGURATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA = os.path.join(BASE, "input_data")
OUTPUT_DATA = os.path.join(BASE, "output_data")

# Create output subdirectories
os.makedirs(OUTPUT_DATA, exist_ok=True)
for subdir in ["figures", "figures/eda", "figures/garch", "figures/fund_flow",
               "figures/portfolio", "figures/efficiency", "figures/rebalancing",
               "figures/summary"]:
    os.makedirs(os.path.join(OUTPUT_DATA, subdir), exist_ok=True)

# Analysis configuration
WINDOW_START = pd.Timestamp("2021-01-04")
WINDOW_END = pd.Timestamp("2025-10-01")
TRAIN_END = "2023-12-31"
FUNDS = ["AKD", "NBP", "NTI"]
COL_NAME = {"AKD": "akd", "NBP": "nbp", "NTI": "nti"}
FUND_COLORS = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
RISK_FREE = 0.105 / 252

# Plotting configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# Report logger
report_lines = []

def log(msg=""):
    """Print to console and save to report."""
    try:
        print(msg)
    except UnicodeEncodeError:
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe_msg)
    report_lines.append(msg)

def parse_mixed_excel_or_datetime(series):
    """Parse either Excel serial dates or already-formatted date strings."""
    numeric = pd.to_numeric(series, errors='coerce')
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask], unit='D', origin='1899-12-30', errors='coerce'
        )

    text_mask = ~numeric_mask
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(series.loc[text_mask], errors='coerce')

    return parsed

def savefig(subdir, name):
    """Save figure to output/figures/subdir/."""
    path = os.path.join(OUTPUT_DATA, "figures", subdir, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  fig → {path}")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def adf_test(series, name, maxlag=None):
    """Simplified Augmented Dickey-Fuller test."""
    s = np.array(series.dropna(), dtype=float)
    if len(s) < 5:
        return None
    dy = np.diff(s)
    ylag = s[:-1]
    X = np.column_stack([np.ones(len(ylag)), ylag])
    beta, _, _, _ = sp_lstsq(X, dy)
    resid = dy - X @ beta
    s2 = resid @ resid / (len(dy) - 2)
    var_b = s2 * np.linalg.inv(X.T @ X)[1, 1]
    t_stat = beta[1] / np.sqrt(max(var_b, 1e-12))
    pval = stats.t.sf(abs(t_stat), df=len(dy)-2) * 2
    return t_stat, pval

def metrics(y_true, y_pred, label=""):
    """Compute RMSE, MAE, R², directional accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    if label:
        log(f"{label:<30} RMSE={rmse:>8.4f} MAE={mae:>8.4f} "
            f"R²={r2:>7.4f} DirAcc={dir_acc:>6.2%}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "dir_acc": dir_acc}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREPROCESSING & DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 1 — DATA PREPROCESSING")
log("=" * 70)

# Define input file paths — assume they are in input_data/ or in 4_claude_model/data/
# Fallback: try both locations
def find_file(name):
    candidates = [
        os.path.join(INPUT_DATA, name),
        os.path.join(BASE, "..", "4_claude_model", "data", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {INPUT_DATA} or fallback location")

KSE30_STOCKS = find_file("kse-30-basic.xlsx")
FUNDS_FILE = find_file("funds_data.xlsx")
MACRO_FILE = find_file("macro_data.xlsx")
CPI_CSV = find_file("cpi.csv")
try:
    KSE30_INDEX = find_file("kse30_index_level.csv")
    INDEX_AVAILABLE = True
except FileNotFoundError:
    INDEX_AVAILABLE = False
    log("WARNING: kse30_index_level.csv not found — index columns skipped")

# ── 1.1 Load KSE-30 stock data ──────────────────────────────────────────────
log("\n1.1 Loading KSE-30 stock data …")
df_stocks = pd.read_excel(KSE30_STOCKS)
df_stocks['Date'] = parse_mixed_excel_or_datetime(df_stocks['Date'])
df_stocks = df_stocks.rename(columns={
    'Date': 'date', 'SYMBOL': 'symbol', 'COMPANY': 'company',
    'PRICE': 'price', 'IDX WT %': 'weight_pct', 'VOLUME': 'volume'
})
pre = len(df_stocks)
df_stocks = df_stocks[df_stocks['weight_pct'] > 0].copy()
log(f"Dropped {pre - len(df_stocks)} rows with zero weight")

df_stocks = df_stocks.sort_values(['symbol', 'date']).reset_index(drop=True)
df_stocks['log_return'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: np.log(x / x.shift(1)))
)
df_stocks['rolling_vol_30d'] = (
    df_stocks.groupby('symbol')['log_return']
    .transform(lambda x: x.rolling(30, min_periods=15).std() * np.sqrt(252))
)
df_stocks['ma_20'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: x.rolling(20, min_periods=10).mean())
)
df_stocks['ma_50'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: x.rolling(50, min_periods=25).mean())
)

df_stocks = df_stocks[
    (df_stocks['date'] >= WINDOW_START) &
    (df_stocks['date'] <= WINDOW_END)
].copy()

log(f"KSE-30 stocks: {len(df_stocks):,} rows | "
    f"{df_stocks['symbol'].nunique()} unique symbols | "
    f"{df_stocks['date'].min().date()} → {df_stocks['date'].max().date()}")

# ── 1.2 Load KSE-30 index level (if available) ─────────────────────────────
if INDEX_AVAILABLE:
    log("1.2 Loading KSE-30 index level …")
    df_idx = pd.read_csv(KSE30_INDEX, parse_dates=['date'])
    df_idx = df_idx.sort_values('date').reset_index(drop=True)
    df_idx = df_idx.rename(columns={
        'index_return': 'idx_return', 'log_return': 'idx_log_return',
        'total_volume': 'idx_total_volume', 'num_companies': 'idx_num_companies'
    })
    df_idx['idx_rolling_vol_30d'] = (
        df_idx['idx_log_return'].rolling(30, min_periods=15).std() * np.sqrt(252)
    )
    df_idx = df_idx[
        (df_idx['date'] >= WINDOW_START) &
        (df_idx['date'] <= WINDOW_END)
    ].copy()
    log(f"Index level: {len(df_idx):,} rows")
else:
    df_idx = pd.DataFrame()

# ── 1.3 Load fund data ────────────────────────────────────────────────────
log("1.3 Loading fund data (NAV & AUM) …")
funds_daily = {}
funds_monthly = {}

for fund in FUNDS:
    raw = pd.read_excel(FUNDS_FILE, sheet_name=fund)
    raw = raw.rename(columns={'DATE': 'date', 'NAV': 'nav', 'AUM': 'aum'})
    raw = raw.sort_values('date').reset_index(drop=True)

    raw['nav_log_return'] = np.log(raw['nav'] / raw['nav'].shift(1))
    raw['nav_rolling_vol_30d'] = (
        raw['nav_log_return'].rolling(30, min_periods=15).std() * np.sqrt(252)
    )

    raw['month'] = raw['date'].dt.to_period('M')
    monthly = (
        raw.groupby('month')
        .agg(
            date=('date', 'last'), nav_end=('nav', 'last'),
            nav_start=('nav', 'first'), aum=('aum', 'last'),
        )
        .reset_index()
    )

    monthly['aum_prev'] = monthly['aum'].shift(1)
    monthly['nav_return_m'] = monthly['nav_end'] / monthly['nav_start'] - 1
    monthly['fund_flow'] = (
        monthly['aum'] - monthly['aum_prev'] * (1 + monthly['nav_return_m'])
    )
    monthly['fund_flow_pct'] = monthly['fund_flow'] / monthly['aum_prev']
    monthly = monthly.dropna(subset=['fund_flow']).reset_index(drop=True)

    raw_clipped = raw[
        (raw['date'] >= WINDOW_START) & (raw['date'] <= WINDOW_END)
    ].copy()
    monthly_clipped = monthly[
        (monthly['date'] >= WINDOW_START) & (monthly['date'] <= WINDOW_END)
    ].copy()

    funds_daily[fund] = raw_clipped
    funds_monthly[fund] = monthly_clipped
    log(f"{fund}: daily={len(raw_clipped):,} rows | monthly={len(monthly_clipped)} rows")

# ── 1.4 Load macro data ─────────────────────────────────────────────────────
log("1.4 Loading macro data …")

df_oil = pd.read_excel(MACRO_FILE, sheet_name='OIL')
df_oil = df_oil.rename(columns={'DATE': 'date', 'PRICE': 'oil_price'})
df_oil = df_oil.sort_values('date').reset_index(drop=True)
df_oil['oil_log_return'] = np.log(df_oil['oil_price'] / df_oil['oil_price'].shift(1))

df_ir = pd.read_excel(MACRO_FILE, sheet_name='IR')
df_ir = df_ir.rename(columns={'DATE': 'date', 'RATE': 'interest_rate'})
df_ir = df_ir.sort_values('date').reset_index(drop=True)

all_days = pd.date_range(WINDOW_START, WINDOW_END, freq='D')
df_ir_daily = (
    df_ir.set_index('date')
    .reindex(all_days)
    .rename_axis('date')
    .reset_index()
)
df_ir_daily['interest_rate'] = df_ir_daily['interest_rate'].ffill().bfill()

df_usd = pd.read_excel(MACRO_FILE, sheet_name='USD')
df_usd = df_usd.rename(columns={'DATE': 'date', 'USD': 'usdpkr'})
df_usd = df_usd.sort_values('date').reset_index(drop=True)
df_usd['usdpkr_log_return'] = np.log(df_usd['usdpkr'] / df_usd['usdpkr'].shift(1))

df_cpi = pd.read_csv(CPI_CSV, skiprows=1, header=0)
df_cpi.columns = ['period_str', 'cpi_yoy']
df_cpi = df_cpi.dropna()
df_cpi['cpi_yoy'] = pd.to_numeric(df_cpi['cpi_yoy'], errors='coerce')
df_cpi = df_cpi.dropna(subset=['cpi_yoy'])
df_cpi['date'] = pd.to_datetime(df_cpi['period_str'], format='%b-%y', errors='coerce')
df_cpi = df_cpi[df_cpi['date'].notna()].copy()
df_cpi['date'] = df_cpi['date'] + pd.offsets.MonthEnd(0)
df_cpi = df_cpi[['date', 'cpi_yoy']].sort_values('date').reset_index(drop=True)

df_cpi_daily = (
    df_cpi.set_index('date')
    .reindex(all_days)
    .rename_axis('date')
    .reset_index()
)
df_cpi_daily['cpi_yoy'] = df_cpi_daily['cpi_yoy'].ffill().bfill()

log(f"Oil: {len(df_oil):,} rows | IR: {len(df_ir)} decisions | "
    f"USD: {len(df_usd):,} rows | CPI: {len(df_cpi)} months")

# ── 1.5 Build daily master table ────────────────────────────────────────────
log("1.5 Building daily master table …")

trading_days = df_stocks['date'].drop_duplicates().sort_values().reset_index(drop=True)
daily = pd.DataFrame({'date': trading_days})

if INDEX_AVAILABLE:
    daily = daily.merge(df_idx, on='date', how='left')

daily = daily.merge(df_oil[['date', 'oil_price', 'oil_log_return']], on='date', how='left')
daily = daily.merge(df_usd[['date', 'usdpkr', 'usdpkr_log_return']], on='date', how='left')
daily = daily.merge(df_ir_daily[['date', 'interest_rate']], on='date', how='left')
daily = daily.merge(df_cpi_daily[['date', 'cpi_yoy']], on='date', how='left')

for fund, df_f in funds_daily.items():
    f = fund.lower()
    df_f_renamed = df_f[['date', 'nav', 'nav_log_return', 'nav_rolling_vol_30d']].copy()
    df_f_renamed.columns = ['date', f'nav_{f}', f'nav_return_{f}', f'nav_vol_{f}']
    daily = daily.merge(df_f_renamed, on='date', how='left')

macro_ffill_cols = ['oil_price', 'oil_log_return', 'usdpkr', 'usdpkr_log_return',
                    'interest_rate', 'cpi_yoy']
daily[macro_ffill_cols] = daily[macro_ffill_cols].ffill()

nav_cols = [c for c in daily.columns if c.startswith('nav_')]
daily[nav_cols] = daily[nav_cols].ffill()

log(f"Daily master: {len(daily):,} rows × {len(daily.columns)} columns")

# ── 1.6 Build monthly master table ──────────────────────────────────────────
log("1.6 Building monthly master table …")

daily['month'] = daily['date'].dt.to_period('M')

agg_dict = {
    'date': ('date', 'last'),
    'oil_price_end': ('oil_price', 'last'),
    'oil_return_monthly': ('oil_log_return', 'sum'),
    'usdpkr_end': ('usdpkr', 'last'),
    'usdpkr_return_monthly': ('usdpkr_log_return', 'sum'),
    'interest_rate_end': ('interest_rate', 'last'),
    'cpi_yoy_end': ('cpi_yoy', 'last'),
}

if INDEX_AVAILABLE:
    agg_dict.update({
        'idx_return_monthly': ('idx_log_return', 'sum'),
        'idx_vol_monthly': ('idx_rolling_vol_30d', 'last'),
        'idx_total_vol': ('idx_total_volume', 'sum'),
    })

monthly_agg = daily.groupby('month').agg(**agg_dict).reset_index(drop=True)

for fund, df_m in funds_monthly.items():
    f = fund.lower()
    df_m_renamed = df_m[[
        'date', 'nav_end', 'nav_return_m', 'aum', 'fund_flow', 'fund_flow_pct'
    ]].copy()
    df_m_renamed.columns = [
        'date', f'nav_{f}_end', f'nav_return_{f}_monthly', f'aum_{f}',
        f'flow_{f}', f'flow_pct_{f}'
    ]
    monthly_agg = monthly_agg.merge(df_m_renamed, on='date', how='left')

monthly_agg = monthly_agg.dropna(
    subset=[c for c in monthly_agg.columns if c.startswith('flow_')]
).reset_index(drop=True)

flow_cols = [c for c in monthly_agg.columns if (c.startswith('flow_') and
                                                 not c.endswith('_pct'))]
monthly_agg['total_fund_flow'] = monthly_agg[flow_cols].sum(axis=1)

log(f"Monthly master: {len(monthly_agg):,} rows × {len(monthly_agg.columns)} columns")
log(f"Date range: {monthly_agg['date'].min().date()} → {monthly_agg['date'].max().date()}")

# Save masters to output_data/
daily.to_csv(os.path.join(OUTPUT_DATA, 'daily_master.csv'), index=False)
monthly_agg.to_csv(os.path.join(OUTPUT_DATA, 'monthly_master.csv'), index=False)
df_stocks.to_csv(os.path.join(OUTPUT_DATA, 'kse30_stocks_daily.csv'), index=False)
log(f"✓ Saved masters to {OUTPUT_DATA}/")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 2 — EXPLORATORY DATA ANALYSIS")
log("=" * 70)

# ── 2.1 Daily macro overview ────────────────────────────────────────────────
log("2.1 Creating daily macro time-series figure …")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Daily Macro Indicators — KSE-30 Analysis (2021–2025)", fontweight="bold")

ax = axes[0]
ax.plot(daily["date"], daily["oil_price"], color="#c0392b", linewidth=1)
ax.set_ylabel("Brent Oil (USD/bbl)")
ax.set_title("Oil Price")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(daily["date"], daily["usdpkr"], color="#8e44ad", linewidth=1)
ax.set_ylabel("PKR per USD")
ax.set_title("USD/PKR Exchange Rate")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(daily["date"], daily["interest_rate"], color="#27ae60", linewidth=1.5, label='KIBOR')
ax.set_ylabel("Interest Rate (%)")
ax.set_xlabel("Date")
ax.set_title("Policy Rate (KIBOR)")
ax.grid(True, alpha=0.3)
ax.legend()

savefig("eda", "01_macro_overview.png")

# ── 2.2 Fund AUM trends ─────────────────────────────────────────────────────
log("2.2 Creating AUM trends figure …")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Fund AUM Trends", fontweight="bold")

for i, fund in enumerate(FUNDS):
    f = fund.lower()
    col = f'aum_{f}'
    if col in monthly_agg.columns:
        ax = axes[i]
        ax.plot(monthly_agg['date'], monthly_agg[col], color=FUND_COLORS[fund],
                linewidth=2, marker='o', markersize=3)
        ax.set_ylabel(f"{fund} AUM (PKR mn)")
        ax.set_title(f"{fund} Assets Under Management")
        ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Date")
savefig("eda", "02_aum_trends.png")

# ── 2.3 Fund flows ──────────────────────────────────────────────────────────
log("2.3 Creating fund flows figure …")
fig, ax = plt.subplots(figsize=(13, 6))
colors_neg = ['#d62728' if x < 0 else '#2ca02c' for x in monthly_agg['total_fund_flow']]
ax.bar(monthly_agg['date'], monthly_agg['total_fund_flow'], color=colors_neg, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Fund Flow (PKR mn)")
ax.set_title("Aggregate Monthly Fund Flows (AKD + NBP + NTI)")
ax.grid(True, alpha=0.3, axis='y')
savefig("eda", "03_total_flows.png")

# ── 2.4 Correlation heatmap ─────────────────────────────────────────────────
log("2.4 Creating correlation heatmap …")
corr_cols = [
    'total_fund_flow', 'oil_price_end', 'usdpkr_end',
    'interest_rate_end', 'cpi_yoy_end'
]
if INDEX_AVAILABLE and 'idx_return_monthly' in monthly_agg.columns:
    corr_cols.append('idx_return_monthly')

corr_data = monthly_agg[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title("Monthly Correlations")
savefig("eda", "04_correlation_heatmap.png")

log("✓ EDA figures completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUND FLOW PREDICTION (ARIMAX + VAR)
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 3 — FUND FLOW PREDICTION (ARIMAX + VAR)")
log("=" * 70)

# Prepare data: split into train/test
monthly_agg['year_month'] = monthly_agg['date'].dt.to_period('M')
train_mask = monthly_agg['date'] <= TRAIN_END
test_mask = ~train_mask

X_train = monthly_agg.loc[train_mask].copy()
X_test = monthly_agg.loc[test_mask].copy()

y_train = X_train['total_fund_flow'].values
y_test = X_test['total_fund_flow'].values

log(f"Train: {len(X_train)} months | Test: {len(X_test)} months")

# ── 3.1 Naïve baseline (last value forward) ─────────────────────────────────
log("3.1 Naïve baseline (last value forward) …")
naive_pred = np.full_like(y_test, y_train[-1], dtype=float)
metrics(y_test, naive_pred, "Naïve baseline")

# ── 3.2 Simple AR(1) ────────────────────────────────────────────────────────
log("3.2 Simple AR(1) model …")
# AR(1): y_t = c + φ₁*y_{t-1} + ε_t
X_ar = np.column_stack([np.ones(len(y_train)-1), y_train[:-1]])
y_ar = y_train[1:]
beta_ar = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]

ar_pred = []
last_val = y_train[-1]
for _ in range(len(y_test)):
    next_val = beta_ar[0] + beta_ar[1] * last_val
    ar_pred.append(next_val)
    last_val = next_val
ar_pred = np.array(ar_pred)
metrics(y_test, ar_pred, "AR(1) model")

# ── 3.3 ARIMAX with macro regressors ────────────────────────────────────────
log("3.3 ARIMAX(1,0,1) with macro exogenous regressors …")

# Features: lagged flow + macro
macro_feat = ['oil_return_monthly', 'usdpkr_return_monthly', 'interest_rate_end', 'cpi_yoy_end']
macro_train = X_train[macro_feat].fillna(0).values
macro_test = X_test[macro_feat].fillna(0).values

# Standardise
scaler = StandardScaler()
macro_train_std = scaler.fit_transform(macro_train)
macro_test_std = scaler.transform(macro_test)

# Build feature matrix for training
X_arimax = np.column_stack([
    np.ones(len(y_train)-1),
    y_train[:-1],  # lagged flow
    macro_train_std[:-1],  # contemporaneous macro
])

y_arimax = y_train[1:]
beta_arimax = np.linalg.lstsq(X_arimax, y_arimax, rcond=None)[0]

# Forecast
arimax_pred = []
last_flow = y_train[-1]
for i in range(len(y_test)):
    X_i = np.append(1, [last_flow] + macro_test_std[i].tolist())
    next_flow = X_i @ beta_arimax
    arimax_pred.append(next_flow)
    last_flow = next_flow
arimax_pred = np.array(arimax_pred)
metrics(y_test, arimax_pred, "ARIMAX(1,0,1)")

# ── 3.4 Summary table ───────────────────────────────────────────────────────
results_fund_flow = pd.DataFrame({
    'Model': ['Naïve', 'AR(1)', 'ARIMAX(1,0,1)'],
    'RMSE': [metrics(y_test, naive_pred)['rmse'],
             metrics(y_test, ar_pred)['rmse'],
             metrics(y_test, arimax_pred)['rmse']],
    'MAE': [metrics(y_test, naive_pred)['mae'],
            metrics(y_test, ar_pred)['mae'],
            metrics(y_test, arimax_pred)['mae']],
    'R²': [metrics(y_test, naive_pred)['r2'],
           metrics(y_test, ar_pred)['r2'],
           metrics(y_test, arimax_pred)['r2']],
})

log("\nFund flow model summary:")
log(results_fund_flow.to_string(index=False))
results_fund_flow.to_csv(os.path.join(OUTPUT_DATA, 'results_fund_flow.csv'), index=False)

# ── 3.5 Plot predictions ────────────────────────────────────────────────────
log("3.5 Plotting fund flow predictions …")
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(X_train['date'], y_train, color='black', linewidth=2, label='Train data', marker='o', markersize=3)
ax.plot(X_test['date'], y_test, color='blue', linewidth=2, label='Actual test', marker='o', markersize=3)
ax.plot(X_test['date'], arimax_pred, color='red', linewidth=2, linestyle='--', label='ARIMAX forecast', marker='s', markersize=3)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel("Date")
ax.set_ylabel("Fund Flow (PKR mn)")
ax.set_title("Fund Flow Prediction: ARIMAX(1,0,1) vs Actuals")
ax.legend()
ax.grid(True, alpha=0.3)
savefig("fund_flow", "01_flow_predictions.png")

log("✓ Fund flow analysis completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — GARCH VOLATILITY MODELLING
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 4 — GARCH VOLATILITY MODELLING")
log("=" * 70)

def garch_nll(params, returns):
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or (alpha + beta) >= 1:
        return 1e10
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        if sigma2[t] <= 0:
            return 1e10
    nll = 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    return nll

results_garch = []

for fund in FUNDS:
    f = fund.lower()
    col = f'nav_return_{f}'
    if col in daily.columns:
        log(f"Fitting GARCH(1,1) for {fund} …")
        returns = daily[col].dropna().values * 100  # in %

        # Fit GARCH(1,1)
        x0 = [np.var(returns) * 0.01, 0.05, 0.9]
        result = minimize(garch_nll, x0, args=(returns,), method='Nelder-Mead')
        omega, alpha, beta = result.x
        persistence = alpha + beta

        results_garch.append({
            'Fund': fund,
            'Model': 'GARCH(1,1)',
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'Persistence': persistence,
            'LL': -garch_nll(result.x, returns),
        })

        log(f"  ω={omega:.6f}, α={alpha:.6f}, β={beta:.6f}, persistence={persistence:.4f}")

results_garch_df = pd.DataFrame(results_garch)
results_garch_df.to_csv(os.path.join(OUTPUT_DATA, 'results_garch.csv'), index=False)

log(f"\n{results_garch_df.to_string(index=False)}")
log("✓ GARCH modelling completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PORTFOLIO OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 5 — PORTFOLIO OPTIMISATION")
log("=" * 70)

log("5.1 Building stock-level return matrix …")

# Wide-form returns (stocks × time)
stocks_pivot = df_stocks.pivot_table(
    index='date', columns='symbol', values='log_return', aggfunc='last'
)
stocks_pivot = stocks_pivot.dropna(axis=0, how='all').dropna(axis=1, how='all')
log(f"Stock returns: {stocks_pivot.shape[0]} dates × {stocks_pivot.shape[1]} stocks")

# Correlation matrix
corr_matrix = stocks_pivot.corr()
log(f"Correlation matrix: {corr_matrix.shape}")

# Average returns and covariance
avg_returns = stocks_pivot.mean()
cov_matrix = stocks_pivot.cov()

# Mean-variance optimisation
def portfolio_metrics(weights, returns, cov, rf):
    """Portfolio return, volatility, Sharpe ratio."""
    port_return = (weights * returns).sum()
    port_vol = np.sqrt(weights @ cov @ weights)
    sharpe = (port_return - rf) / max(port_vol, 1e-6)
    return port_return, port_vol, sharpe

def neg_sharpe(weights, returns, cov, rf):
    """Negative Sharpe (for minimisation)."""
    _, _, sharpe = portfolio_metrics(weights, returns, cov, rf)
    return -sharpe

# Equal-weight portfolio
n_stocks = len(avg_returns)
w_equal = np.ones(n_stocks) / n_stocks
ret_eq, vol_eq, sharpe_eq = portfolio_metrics(w_equal, avg_returns, cov_matrix, RISK_FREE)

# Market-cap weight (use average index weight)
weights_mcap = df_stocks.groupby('symbol')['weight_pct'].mean().values
weights_mcap = weights_mcap / weights_mcap.sum()
ret_mc, vol_mc, sharpe_mc = portfolio_metrics(weights_mcap, avg_returns, cov_matrix, RISK_FREE)

# Max Sharpe portfolio
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(n_stocks))
result_max_sharpe = minimize(neg_sharpe, w_equal, args=(avg_returns, cov_matrix, RISK_FREE),
                             method='SLSQP', bounds=bounds, constraints=constraints)
w_max_sharpe = result_max_sharpe.x
ret_ms, vol_ms, sharpe_ms = portfolio_metrics(w_max_sharpe, avg_returns, cov_matrix, RISK_FREE)

log(f"\nPortfolio comparison:")
log(f"  Equal-weight:    ret={ret_eq:.4f}  vol={vol_eq:.4f}  Sharpe={sharpe_eq:.4f}")
log(f"  Market-cap:      ret={ret_mc:.4f}  vol={vol_mc:.4f}  Sharpe={sharpe_mc:.4f}")
log(f"  Max-Sharpe:      ret={ret_ms:.4f}  vol={vol_ms:.4f}  Sharpe={sharpe_ms:.4f}")

# Plot efficient frontier
fig, ax = plt.subplots(figsize=(10, 7))

# Random portfolios for frontier approximation
np.random.seed(42)
n_portfolios = 5000
results_rand = np.zeros((3, n_portfolios))
for i in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n_stocks))
    ret, vol, sharpe = portfolio_metrics(w, avg_returns, cov_matrix, RISK_FREE)
    results_rand[0, i] = vol
    results_rand[1, i] = ret
    results_rand[2, i] = sharpe

scatter = ax.scatter(results_rand[0], results_rand[1], c=results_rand[2], cmap='viridis',
                     alpha=0.5, s=10, label='Random portfolios')
ax.scatter([vol_eq], [ret_eq], color='red', s=200, marker='s', label='Equal-weight', zorder=5)
ax.scatter([vol_mc], [ret_mc], color='orange', s=200, marker='^', label='Market-cap', zorder=5)
ax.scatter([vol_ms], [ret_ms], color='green', s=200, marker='*', label='Max Sharpe', zorder=5)

ax.set_xlabel("Volatility (std)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier: KSE-30 Stocks")
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
savefig("portfolio", "01_efficient_frontier.png")

# Save portfolio weights
weights_df = pd.DataFrame({
    'Symbol': avg_returns.index,
    'Equal_Weight': w_equal,
    'MarketCap_Weight': weights_mcap,
    'MaxSharpe_Weight': w_max_sharpe,
})
weights_df.to_csv(os.path.join(OUTPUT_DATA, 'portfolio_weights.csv'), index=False)

log("✓ Portfolio optimisation completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — MARKET EFFICIENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 6 — MARKET EFFICIENCY ANALYSIS")
log("=" * 70)

log("6.1 Runs test (randomness of return signs) …")

def runs_test(returns):
    """Non-parametric test: count runs of positive vs negative returns."""
    signs = np.sign(returns)
    n_pos = np.sum(signs == 1)
    n_neg = np.sum(signs == -1)
    n_total = n_pos + n_neg
    
    runs = 1 + np.sum(np.diff(signs) != 0)
    mean_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
    var_runs = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg) / (
        (n_pos + n_neg)**2 * (n_pos + n_neg - 1)
    )
    z_stat = (runs - mean_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return runs, z_stat, p_value

efficiency_results = []

# Daily NAV returns
for fund in FUNDS:
    f = fund.lower()
    col = f'nav_return_{f}'
    if col in daily.columns:
        returns = daily[col].dropna().values
        runs, z, pval = runs_test(returns)
        log(f"{fund} NAV returns: runs={runs}, Z={z:.4f}, p={pval:.4f}")
        efficiency_results.append({
            'Asset': f'{fund} NAV',
            'Test': 'Runs',
            'Statistic': z,
            'P_Value': pval,
            'Result': 'Random' if pval > 0.05 else 'Non-random',
        })

# Stock-level runs tests
log("\nStock-level efficiency (sample of top stocks):")
for symbol in df_stocks['symbol'].unique()[:5]:
    stock_data = df_stocks[df_stocks['symbol'] == symbol].sort_values('date')
    returns = stock_data['log_return'].dropna().values
    if len(returns) > 30:
        runs, z, pval = runs_test(returns)
        log(f"{symbol:8s}: runs={runs:4d}, Z={z:7.4f}, p={pval:.4f}")
        efficiency_results.append({
            'Asset': symbol,
            'Test': 'Runs',
            'Statistic': z,
            'P_Value': pval,
            'Result': 'Random' if pval > 0.05 else 'Non-random',
        })

# Variance ratio test
log("6.2 Variance ratio test (random walk hypothesis) …")

def variance_ratio(returns, q=2):
    """Variance ratio: Var(qΔX) / q*Var(ΔX)."""
    n = len(returns)
    var_1 = np.var(returns, ddof=1)
    
    returns_q = []
    for i in range(0, n-q+1, q):
        returns_q.append(np.sum(returns[i:i+q]))
    var_q = np.var(returns_q, ddof=1)
    
    vr = var_q / (q * var_1) if var_1 > 0 else np.nan
    return vr

for fund in FUNDS:
    f = fund.lower()
    col = f'nav_return_{f}'
    if col in daily.columns:
        returns = daily[col].dropna().values
        vr2 = variance_ratio(returns, q=2)
        vr4 = variance_ratio(returns, q=4)
        log(f"{fund}: VR(2)={vr2:.4f}, VR(4)={vr4:.4f}")

efficiency_df = pd.DataFrame(efficiency_results)
efficiency_df.to_csv(os.path.join(OUTPUT_DATA, 'results_efficiency.csv'), index=False)

# Plot ACF (autocorrelation)
log("6.3 Plotting ACF (autocorrelation) …")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, fund in enumerate(FUNDS):
    f = fund.lower()
    col = f'nav_return_{f}'
    if col in daily.columns:
        returns = daily[col].dropna().values
        acf_vals = [np.corrcoef(returns[:-j], returns[j:])[0, 1] for j in range(1, 21)]
        
        ax = axes[i]
        ax.bar(range(1, 21), acf_vals, color=FUND_COLORS[fund], alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(1.96/np.sqrt(len(returns)), color='red', linestyle='--', label='95% CI')
        ax.axhline(-1.96/np.sqrt(len(returns)), color='red', linestyle='--')
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_title(f"{fund} NAV Returns ACF")
        ax.legend()
        ax.grid(True, alpha=0.3)

savefig("efficiency", "01_acf.png")

log("✓ Efficiency analysis completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — REBALANCING PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 7 — REBALANCING PREDICTION")
log("=" * 70)

log("7.1 Identifying rebalancing dates …")

# Known KSE-30 rebalancing dates (semi-annual: ~March 15, ~September 15)
rebalancing_dates = [
    pd.Timestamp('2021-03-15'),
    pd.Timestamp('2021-09-15'),
    pd.Timestamp('2022-03-15'),
    pd.Timestamp('2022-09-15'),
    pd.Timestamp('2023-03-15'),
    pd.Timestamp('2023-09-15'),
    pd.Timestamp('2024-03-15'),
    pd.Timestamp('2024-09-16'),
    pd.Timestamp('2025-03-17'),
]

log(f"Rebalancing dates: {[d.date() for d in rebalancing_dates]}")

# Build rebalancing panel: features 30 days before, target on rebalancing date
log("7.2 Building rebalancing feature panel …")

rebal_panel = []
for rebal_date in rebalancing_dates:
    window_start = rebal_date - pd.Timedelta(days=30)
    window_data = df_stocks[
        (df_stocks['date'] >= window_start) &
        (df_stocks['date'] < rebal_date)
    ].copy()

    rebal_data = df_stocks[df_stocks['date'] == rebal_date].copy()

    for symbol in window_data['symbol'].unique():
        stock_window = window_data[window_data['symbol'] == symbol]
        
        if len(stock_window) < 5:
            continue

        # Features
        momentum = stock_window['log_return'].mean() * 252
        volatility = stock_window['log_return'].std() * np.sqrt(252)
        avg_weight = stock_window['weight_pct'].mean()
        avg_volume = stock_window['volume'].mean()

        # Target: is stock still in index at rebalancing?
        in_index = len(rebal_data[rebal_data['symbol'] == symbol]) > 0
        new_weight = rebal_data[rebal_data['symbol'] == symbol]['weight_pct'].values
        new_weight = new_weight[0] if len(new_weight) > 0 else np.nan

        rebal_panel.append({
            'rebalancing_date': rebal_date,
            'symbol': symbol,
            'momentum': momentum,
            'volatility': volatility,
            'avg_weight': avg_weight,
            'avg_volume': avg_volume,
            'retained': in_index,
            'new_weight': new_weight,
        })

rebal_df = pd.DataFrame(rebal_panel)
log(f"Rebalancing panel: {len(rebal_df)} stock-rebalancing pairs")

# Train/test split
train_rebal_dates = rebalancing_dates[:7]
test_rebal_dates = rebalancing_dates[7:]

train_rebal = rebal_df[rebal_df['rebalancing_date'].isin(train_rebal_dates)].copy()
test_rebal = rebal_df[rebal_df['rebalancing_date'].isin(test_rebal_dates)].copy()

log(f"Train rebalanc: {len(train_rebal)} pairs | Test: {len(test_rebal)} pairs")

# ── 7.3 Inclusion prediction (classification) ──────────────────────────────
log("7.3 Training inclusion predictor (Ridge logistic regression) …")

X_rebal_train = train_rebal[['momentum', 'volatility', 'avg_weight', 'avg_volume']].fillna(0)
y_rebal_train = train_rebal['retained'].astype(int)

scaler_rebal = StandardScaler()
X_rebal_train_std = scaler_rebal.fit_transform(X_rebal_train)

logreg = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True)
logreg.fit(X_rebal_train_std, y_rebal_train)

# Test
if len(test_rebal) > 0:
    X_rebal_test = test_rebal[['momentum', 'volatility', 'avg_weight', 'avg_volume']].fillna(0)
    X_rebal_test_std = scaler_rebal.transform(X_rebal_test)
    y_rebal_test = test_rebal['retained'].astype(int)
    
    y_pred_rebal = logreg.predict(X_rebal_test_std)
    acc_rebal = accuracy_score(y_rebal_test, y_pred_rebal)
    log(f"Inclusion prediction accuracy: {acc_rebal:.2%}")

# ── 7.4 Weight prediction (regression) ─────────────────────────────────────
log("7.4 Training weight predictor (Ridge regression) …")

train_rebal_weight = train_rebal.dropna(subset=['new_weight'])
X_weight_train = train_rebal_weight[
    ['momentum', 'volatility', 'avg_weight', 'avg_volume']
].fillna(0)
y_weight_train = train_rebal_weight['new_weight']

X_weight_train_std = scaler_rebal.fit_transform(X_weight_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_weight_train_std, y_weight_train)

# Test
test_rebal_weight = test_rebal.dropna(subset=['new_weight'])
if len(test_rebal_weight) > 0:
    X_weight_test = test_rebal_weight[
        ['momentum', 'volatility', 'avg_weight', 'avg_volume']
    ].fillna(0)
    X_weight_test_std = scaler_rebal.transform(X_weight_test)
    y_weight_test = test_rebal_weight['new_weight']
    
    y_pred_weight = ridge.predict(X_weight_test_std)
    metrics(y_weight_test, y_pred_weight, "Weight prediction (Ridge)")

rebal_df.to_csv(os.path.join(OUTPUT_DATA, 'results_rebalancing.csv'), index=False)

log("✓ Rebalancing analysis completed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

log("\n" + "=" * 70)
log("SECTION 8 — RESULTS SUMMARY")
log("=" * 70)

# Create summary dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Fund flows
ax1 = fig.add_subplot(gs[0, :2])
colors_flow = ['#d62728' if x < 0 else '#2ca02c' for x in monthly_agg['total_fund_flow']]
ax1.bar(monthly_agg['date'], monthly_agg['total_fund_flow'], color=colors_flow, alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylabel("Flow (PKR mn)")
ax1.set_title("Total Fund Flows")
ax1.grid(True, alpha=0.3, axis='y')

# 2. AUM trends
ax2 = fig.add_subplot(gs[0, 2])
for fund in FUNDS:
    f = fund.lower()
    col = f'aum_{f}'
    if col in monthly_agg.columns:
        ax2.plot(monthly_agg['date'], monthly_agg[col], label=fund, color=FUND_COLORS[fund])
ax2.set_ylabel("AUM (PKR mn)")
ax2.set_title("Fund AUM Trends")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Oil price
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(daily['date'], daily['oil_price'], color='#c0392b', linewidth=1)
ax3.set_ylabel("Oil (USD/bbl)")
ax3.set_title("Brent Oil Price")
ax3.grid(True, alpha=0.3)

# 4. USD/PKR
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(daily['date'], daily['usdpkr'], color='#8e44ad', linewidth=1)
ax4.set_ylabel("PKR/USD")
ax4.set_title("USD/PKR Exchange Rate")
ax4.grid(True, alpha=0.3)

# 5. Interest rate
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(daily['date'], daily['interest_rate'], color='#27ae60', linewidth=1.5)
ax5.set_ylabel("Rate (%)")
ax5.set_title("Policy Rate (KIBOR)")
ax5.grid(True, alpha=0.3)

# 6. Model performance
ax6 = fig.add_subplot(gs[2, :])
models = results_fund_flow['Model'].values
rmse_vals = results_fund_flow['RMSE'].values
x_pos = np.arange(len(models))
ax6.bar(x_pos, rmse_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(models)
ax6.set_ylabel("RMSE (PKR mn)")
ax6.set_title("Fund Flow Model Comparison")
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle("Summary Dashboard — KSE-30 Fund Flow & Market Analysis", fontweight="bold", fontsize=14)
savefig("summary", "01_dashboard.png")

# Save report
report_path = os.path.join(OUTPUT_DATA, 'pipeline_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"\n✓ Report saved to {report_path}")

log("\n" + "=" * 70)
log("PIPELINE COMPLETED SUCCESSFULLY")
log("=" * 70)
log(f"All outputs saved to: {OUTPUT_DATA}/")
log("\nOutput files:")
log(f"  • daily_master.csv")
log(f"  • monthly_master.csv")
log(f"  • kse30_stocks_daily.csv")
log(f"  • portfolio_weights.csv")
log(f"  • results_fund_flow.csv")
log(f"  • results_garch.csv")
log(f"  • results_efficiency.csv")
log(f"  • results_rebalancing.csv")
log(f"  • figures/ (subfolder with 8+ PNG figures)")
log(f"  • pipeline_report.txt")
log("=" * 70)
