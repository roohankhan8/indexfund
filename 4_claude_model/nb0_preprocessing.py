"""
FYDP Preprocessing Pipeline
============================
Analyzing Fund Flow Patterns and Market Efficiency in Pakistan
KSE-30 | AKD, NBP, NIT funds | Macro indicators

Output files (written to same directory as this script):
  - daily_master.csv      → aligned daily dataset (common window)
  - monthly_master.csv    → aligned monthly dataset with fund flows
  - kse30_stocks_daily.csv → per-stock long-format daily data
  - preprocessing_report.txt → data quality and summary report

Analysis window: 2021-01-04 → 2025-10-01
  (start = first date all daily data overlaps: AKD Jan 2021, OIL Jan 2021)
  (end   = last KSE-30 stock data date)

Run: python preprocessing.py
Requirements: pandas, numpy, openpyxl, scipy
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = BASE  # adjust if data files are in a subfolder

KSE30_STOCKS  = os.path.join(DATA, "data/kse-30-basic.xlsx")
FUNDS         = os.path.join(DATA, "data/funds_data.xlsx")
MACRO         = os.path.join(DATA, "data/macro_data.xlsx")
CPI_CSV       = os.path.join(DATA, "data/cpi.csv")
KSE30_INDEX   = os.path.join(DATA, "data/kse30_index_level.csv")

# ── analysis window ────────────────────────────────────────────────────────
WINDOW_START = pd.Timestamp("2021-01-04")
WINDOW_END   = pd.Timestamp("2025-10-01")

# ── report collector ───────────────────────────────────────────────────────
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — KSE-30 STOCK DATA (per-company daily)
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 1 — KSE-30 stock data")
log("=" * 65)

df_stocks = pd.read_excel(KSE30_STOCKS)

# Fix Excel serial date format
df_stocks['Date'] = pd.to_datetime(
    df_stocks['Date'], unit='D', origin='1899-12-30'
)
df_stocks = df_stocks.rename(columns={
    'Date':     'date',
    'SYMBOL':   'symbol',
    'COMPANY':  'company',
    'PRICE':    'price',
    'IDX WT %': 'weight_pct',
    'VOLUME':   'volume'
})

# Drop the 120 rows where weight = 0 (stock was not in index that day)
pre = len(df_stocks)
df_stocks = df_stocks[df_stocks['weight_pct'] > 0].copy()
log(f"Dropped {pre - len(df_stocks)} rows with weight = 0 (non-constituent days)")

# Sort
df_stocks = df_stocks.sort_values(['symbol', 'date']).reset_index(drop=True)

# Per-stock log return
df_stocks['log_return'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: np.log(x / x.shift(1)))
)

# Per-stock 30-day rolling volatility (annualised: ×√252)
df_stocks['rolling_vol_30d'] = (
    df_stocks.groupby('symbol')['log_return']
    .transform(lambda x: x.rolling(30, min_periods=15).std() * np.sqrt(252))
)

# Per-stock 20-day and 50-day moving average of price
df_stocks['ma_20'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: x.rolling(20, min_periods=10).mean())
)
df_stocks['ma_50'] = (
    df_stocks.groupby('symbol')['price']
    .transform(lambda x: x.rolling(50, min_periods=25).mean())
)

# Clip to analysis window
df_stocks = df_stocks[
    (df_stocks['date'] >= WINDOW_START) &
    (df_stocks['date'] <= WINDOW_END)
].copy()

log(f"KSE-30 stocks: {len(df_stocks):,} rows | "
    f"{df_stocks['symbol'].nunique()} unique symbols | "
    f"{df_stocks['date'].min().date()} → {df_stocks['date'].max().date()}")
log(f"Symbols present: {sorted(df_stocks['symbol'].unique())}")
log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — KSE-30 INDEX LEVEL (daily)
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 2 — KSE-30 index level")
log("=" * 65)

try:
    df_idx = pd.read_csv(KSE30_INDEX, parse_dates=['date'])
    df_idx = df_idx.sort_values('date').reset_index(drop=True)

    # Rename for clarity
    df_idx = df_idx.rename(columns={
        'index_return':     'idx_return',
        'log_return':       'idx_log_return',
        'total_volume':     'idx_total_volume',
        'avg_weight_change':'idx_avg_wt_change',
        'num_companies':    'idx_num_companies'
    })

    # Rolling 30-day index volatility (annualised)
    df_idx['idx_rolling_vol_30d'] = (
        df_idx['idx_log_return']
        .rolling(30, min_periods=15).std() * np.sqrt(252)
    )

    # Clip to window
    df_idx = df_idx[
        (df_idx['date'] >= WINDOW_START) &
        (df_idx['date'] <= WINDOW_END)
    ].copy()

    log(f"Index level: {len(df_idx):,} rows | "
        f"{df_idx['date'].min().date()} → {df_idx['date'].max().date()}")
    log(f"Nulls: {df_idx.isnull().sum().to_dict()}")
    INDEX_AVAILABLE = True

except FileNotFoundError:
    log("WARNING: kse30_index_level.csv not found — index columns will be "
        "skipped. Add the file and re-run.")
    df_idx = pd.DataFrame()
    INDEX_AVAILABLE = False

log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUND DATA (NAV daily, AUM monthly → fund flows)
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 3 — Fund data: NAV returns + fund flows")
log("=" * 65)

funds_daily  = {}   # {fund: daily DataFrame}
funds_monthly = {}  # {fund: monthly DataFrame with fund flow}

for fund in ['AKD', 'NBP', 'NTI']:
    raw = pd.read_excel(FUNDS, sheet_name=fund)
    raw = raw.rename(columns={'DATE': 'date', 'NAV': 'nav', 'AUM': 'aum'})
    raw = raw.sort_values('date').reset_index(drop=True)

    # ── daily NAV features ────────────────────────────────────────────────
    raw['nav_log_return'] = np.log(raw['nav'] / raw['nav'].shift(1))
    raw['nav_rolling_vol_30d'] = (
        raw['nav_log_return']
        .rolling(30, min_periods=15).std() * np.sqrt(252)
    )

    # ── monthly fund flow calculation ─────────────────────────────────────
    # Fund Flow = ΔAUM − (AUM_{t-1} × cumulative NAV return over month)
    # We use month-end NAV and AUM values
    raw['month'] = raw['date'].dt.to_period('M')
    monthly = (
        raw.groupby('month')
        .agg(
            date        = ('date', 'last'),   # month-end date
            nav_end     = ('nav',  'last'),
            nav_start   = ('nav',  'first'),
            aum         = ('aum',  'last'),   # AUM is same all month
        )
        .reset_index()
    )

    monthly['aum_prev']      = monthly['aum'].shift(1)
    monthly['nav_return_m']  = monthly['nav_end'] / monthly['nav_start'] - 1

    # Fund Flow (PKR millions) = new money in/out
    # = AUM_t - AUM_{t-1} × (1 + nav_return_this_month)
    monthly['fund_flow']     = (
        monthly['aum'] - monthly['aum_prev'] * (1 + monthly['nav_return_m'])
    )

    # Normalised flow = fund_flow / aum_prev (as % of prior AUM)
    monthly['fund_flow_pct'] = monthly['fund_flow'] / monthly['aum_prev']

    # Flag large flow spikes (>2 std dev) for later investigation
    flow_mean = monthly['fund_flow'].mean()
    flow_std  = monthly['fund_flow'].std()
    monthly['flow_spike'] = (
        (monthly['fund_flow'] - flow_mean).abs() > 2 * flow_std
    )

    # Drop first row (no prior month for flow calc)
    monthly = monthly.dropna(subset=['fund_flow']).reset_index(drop=True)

    spike_count = monthly['flow_spike'].sum()
    log(f"{fund} — daily rows: {len(raw):,} | "
        f"monthly rows: {len(monthly)} | "
        f"flow spikes (>2σ): {spike_count}")
    log(f"  AUM range: {monthly['aum'].min():.2f} → {monthly['aum'].max():.2f} PKR mn")
    log(f"  Fund flow range: {monthly['fund_flow'].min():.2f} → "
        f"{monthly['fund_flow'].max():.2f} PKR mn")
    if spike_count > 0:
        log(f"  Spike months: "
            f"{monthly.loc[monthly['flow_spike'], 'date'].dt.to_period('M').tolist()}")

    # Clip to window
    raw_clipped     = raw[(raw['date'] >= WINDOW_START) &
                          (raw['date'] <= WINDOW_END)].copy()
    monthly_clipped = monthly[(monthly['date'] >= WINDOW_START) &
                               (monthly['date'] <= WINDOW_END)].copy()

    funds_daily[fund]   = raw_clipped
    funds_monthly[fund] = monthly_clipped

log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MACRO DATA
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 4 — Macro data")
log("=" * 65)

# ── Oil (daily) ────────────────────────────────────────────────────────────
df_oil = pd.read_excel(MACRO, sheet_name='OIL')
df_oil = df_oil.rename(columns={'DATE': 'date', 'PRICE': 'oil_price'})
df_oil = df_oil.sort_values('date').reset_index(drop=True)
df_oil['oil_log_return'] = np.log(df_oil['oil_price'] / df_oil['oil_price'].shift(1))

# ── Interest rate (event-driven → forward-fill to daily) ───────────────────
df_ir = pd.read_excel(MACRO, sheet_name='IR')
df_ir = df_ir.rename(columns={'DATE': 'date', 'RATE': 'interest_rate'})
df_ir = df_ir.sort_values('date').reset_index(drop=True)

# Build a continuous daily series by forward-filling
all_days = pd.date_range(WINDOW_START, WINDOW_END, freq='D')
df_ir_daily = (
    df_ir.set_index('date')
    .reindex(all_days)
    .rename_axis('date')
    .reset_index()
)
# Forward fill: each rate applies until the next decision
df_ir_daily['interest_rate'] = df_ir_daily['interest_rate'].ffill()
# Back-fill for any days before the first recorded decision
df_ir_daily['interest_rate'] = df_ir_daily['interest_rate'].bfill()
log(f"Interest rate: forward-filled to {len(df_ir_daily)} daily rows "
    f"from {df_ir['interest_rate'].iloc[0]} → {df_ir['interest_rate'].iloc[-1]}%")

# Monthly change in rate (for monthly master)
df_ir_daily['ir_monthly_change'] = df_ir_daily['interest_rate'].diff()

# ── USD/PKR (daily) ────────────────────────────────────────────────────────
df_usd = pd.read_excel(MACRO, sheet_name='USD')
df_usd = df_usd.rename(columns={'DATE': 'date', 'USD': 'usdpkr'})
df_usd = df_usd.sort_values('date').reset_index(drop=True)
df_usd['usdpkr_log_return'] = np.log(df_usd['usdpkr'] / df_usd['usdpkr'].shift(1))

# ── CPI (monthly → parse "Jul-18" format) ─────────────────────────────────
df_cpi = pd.read_csv(CPI_CSV, skiprows=1, header=0)
df_cpi.columns = ['period_str', 'cpi_yoy']
df_cpi = df_cpi.dropna()
df_cpi['cpi_yoy'] = pd.to_numeric(df_cpi['cpi_yoy'], errors='coerce')
df_cpi = df_cpi.dropna(subset=['cpi_yoy'])
# Parse period strings — mostly "Jul-18" but some use full name e.g. "June-20"
_month_map = {
    'January':'Jan','February':'Feb','March':'Mar','April':'Apr',
    'May':'May','June':'Jun','July':'Jul','August':'Aug',
    'September':'Sep','October':'Oct','November':'Nov','December':'Dec'
}
def _normalise(s):
    for full, abbr in _month_map.items():
        if s.startswith(full):
            return s.replace(full, abbr, 1)
    return s
df_cpi['period_str'] = df_cpi['period_str'].apply(_normalise)
df_cpi['date'] = pd.to_datetime(df_cpi['period_str'], format='%b-%y')
# Use month-end to align with monthly masters
df_cpi['date'] = df_cpi['date'] + pd.offsets.MonthEnd(0)
df_cpi = df_cpi[['date', 'cpi_yoy']].sort_values('date').reset_index(drop=True)
log(f"CPI: {len(df_cpi)} monthly rows | "
    f"{df_cpi['date'].min().date()} → {df_cpi['date'].max().date()}")

# Forward-fill CPI to daily for daily master
df_cpi_daily = (
    df_cpi.set_index('date')
    .reindex(all_days)
    .rename_axis('date')
    .reset_index()
)
df_cpi_daily['cpi_yoy'] = df_cpi_daily['cpi_yoy'].ffill().bfill()

log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DAILY MASTER TABLE
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 5 — Building daily master table")
log("=" * 65)

# Start from KSE-30 trading days (the ground-truth calendar)
trading_days = df_stocks['date'].drop_duplicates().sort_values().reset_index(drop=True)
daily = pd.DataFrame({'date': trading_days})

# Merge index level (if available)
if INDEX_AVAILABLE:
    daily = daily.merge(df_idx, on='date', how='left')

# Merge macro — all daily series, keeping only trading days
for df_macro, cols in [
    (df_oil,     ['date', 'oil_price', 'oil_log_return']),
    (df_usd,     ['date', 'usdpkr', 'usdpkr_log_return']),
    (df_ir_daily,['date', 'interest_rate']),
    (df_cpi_daily,['date', 'cpi_yoy']),
]:
    daily = daily.merge(df_macro[cols], on='date', how='left')

# Merge each fund's daily NAV + rolling vol
for fund, df_f in funds_daily.items():
    cols = ['date',
            f'nav_{fund.lower()}',
            f'nav_return_{fund.lower()}',
            f'nav_vol_{fund.lower()}']
    df_f_renamed = df_f[['date', 'nav', 'nav_log_return', 'nav_rolling_vol_30d']].copy()
    df_f_renamed.columns = ['date'] + cols[1:]
    # Some fund dates may fall on non-trading days — left join keeps only trading days
    daily = daily.merge(df_f_renamed, on='date', how='left')

# Forward-fill macro on any trading days where macro had no entry
# (weekends already excluded; some holidays in macro data may be missing)
macro_ffill_cols = [
    'oil_price', 'oil_log_return',
    'usdpkr', 'usdpkr_log_return',
    'interest_rate', 'cpi_yoy'
]
daily[macro_ffill_cols] = daily[macro_ffill_cols].ffill()

# For NAV columns: fund may be closed on some trading days — ffill is fine
nav_cols = [c for c in daily.columns if c.startswith('nav_')]
daily[nav_cols] = daily[nav_cols].ffill()

log(f"Daily master: {len(daily):,} rows × {len(daily.columns)} columns")
log(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
log(f"Columns: {list(daily.columns)}")
log()

# Null check
null_summary = daily.isnull().sum()
null_summary = null_summary[null_summary > 0]
if len(null_summary):
    log("Remaining nulls in daily master:")
    log(str(null_summary))
else:
    log("No nulls in daily master.")
log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — MONTHLY MASTER TABLE
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 6 — Building monthly master table")
log("=" * 65)

# Aggregate daily → monthly (use last trading day of each month as anchor)
daily['month'] = daily['date'].dt.to_period('M')

monthly_agg = (
    daily.groupby('month')
    .agg(
        date              = ('date', 'last'),
        # Index
        **({'idx_return_monthly': ('idx_log_return', 'sum'),
            'idx_vol_monthly':    ('idx_rolling_vol_30d', 'last'),
            'idx_total_vol':      ('idx_total_volume', 'sum')}
           if INDEX_AVAILABLE else {}),
        # Macro
        oil_price_end     = ('oil_price', 'last'),
        oil_return_monthly= ('oil_log_return', 'sum'),
        usdpkr_end        = ('usdpkr', 'last'),
        usdpkr_return_monthly = ('usdpkr_log_return', 'sum'),
        interest_rate_end = ('interest_rate', 'last'),
        cpi_yoy_end       = ('cpi_yoy', 'last'),
    )
    .reset_index()
)

# Merge each fund's monthly flow + performance
for fund, df_m in funds_monthly.items():
    f = fund.lower()
    df_m_renamed = df_m[[
        'date', 'nav_end', 'nav_return_m', 'aum',
        'fund_flow', 'fund_flow_pct', 'flow_spike'
    ]].copy()
    df_m_renamed.columns = [
        'date',
        f'nav_{f}_end',
        f'nav_return_{f}_monthly',
        f'aum_{f}',
        f'flow_{f}',
        f'flow_pct_{f}',
        f'flow_spike_{f}'
    ]
    # Merge on month-end date
    monthly_agg = monthly_agg.merge(df_m_renamed, on='date', how='left')

# Drop first row (may have incomplete fund data before all funds launched)
monthly_agg = monthly_agg.dropna(
    subset=[c for c in monthly_agg.columns if c.startswith('flow_')]
).reset_index(drop=True)

# Composite flow: sum of all three fund flows (total market-wide fund flow proxy)
flow_cols = [c for c in monthly_agg.columns if c.startswith('flow_akd') or
             c.startswith('flow_nbp') or c.startswith('flow_nti')]
flow_raw = [c for c in flow_cols if not c.endswith('_pct') and not c.endswith('_spike')]
monthly_agg['total_fund_flow'] = monthly_agg[flow_raw].sum(axis=1)

log(f"Monthly master: {len(monthly_agg):,} rows × {len(monthly_agg.columns)} columns")
log(f"Date range: {monthly_agg['date'].min().date()} → {monthly_agg['date'].max().date()}")
log(f"Columns: {list(monthly_agg.columns)}")
log()

null_m = monthly_agg.isnull().sum()
null_m = null_m[null_m > 0]
if len(null_m):
    log("Nulls in monthly master:")
    log(str(null_m))
else:
    log("No nulls in monthly master.")
log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATIONARITY TESTS (ADF) on key series
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 7 — ADF stationarity tests")
log("=" * 65)
log(f"{'Series':<35} {'ADF stat':>10} {'p-value':>10} {'Stationary?':>12}")
log("-" * 70)

from scipy.linalg import lstsq as sp_lstsq

def adf_test(series, name, maxlag=None):
    """
    Simplified ADF: regress Δy_t on y_{t-1} (with constant).
    t-stat on y_{t-1}: significantly negative → stationary.
    p-value is approximate (t-distribution); proper critical values are
    from MacKinnon (1994). Flag as stationary if p < 0.05.
    """
    s = np.array(series.dropna(), dtype=float)
    dy   = np.diff(s)
    ylag = s[:-1]
    X    = np.column_stack([np.ones(len(ylag)), ylag])
    beta, _, _, _ = sp_lstsq(X, dy)
    resid  = dy - X @ beta
    s2     = resid @ resid / (len(dy) - 2)
    var_b  = s2 * np.linalg.inv(X.T @ X)[1, 1]
    t_stat = beta[1] / np.sqrt(max(var_b, 1e-12))
    pval   = stats.t.sf(abs(t_stat), df=len(dy)-2) * 2
    stationary = "YES" if pval < 0.05 else "NO"
    log(f"{name:<35} {t_stat:>10.4f} {pval:>10.4f} {stationary:>12}")

# Daily series
if INDEX_AVAILABLE and 'idx_log_return' in daily.columns:
    adf_test(daily['idx_log_return'],    "KSE-30 index log return")
    adf_test(daily['idx_total_volume'],  "KSE-30 total volume")

adf_test(daily['oil_log_return'],        "Brent oil log return")
adf_test(daily['usdpkr_log_return'],     "USD/PKR log return")
adf_test(daily['oil_price'],             "Brent oil price (level)")
adf_test(daily['usdpkr'],               "USD/PKR (level)")
adf_test(daily['interest_rate'],         "Interest rate (level)")

# Monthly series
if len(monthly_agg) > 0:
    if INDEX_AVAILABLE and 'idx_return_monthly' in monthly_agg.columns:
        adf_test(monthly_agg['idx_return_monthly'], "KSE-30 monthly return")
    adf_test(monthly_agg['cpi_yoy_end'],           "CPI YoY monthly")
    for fund in ['akd', 'nbp', 'nti']:
        col = f'flow_pct_{fund}'
        if col in monthly_agg.columns:
            adf_test(monthly_agg[col], f"Fund flow % ({fund.upper()})", maxlag=6)

log()
log("Interpretation: log returns of prices are typically stationary (I(0)).")
log("Price levels are typically non-stationary (I(1)) — confirmed above.")
log("Use log returns (not levels) as model inputs for ARIMA/LSTM.")
log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — DESCRIPTIVE STATISTICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 8 — Descriptive statistics")
log("=" * 65)

if INDEX_AVAILABLE and 'idx_log_return' in daily.columns:
    log("\nKSE-30 daily log return:")
    s = daily['idx_log_return'].dropna()
    log(f"  Mean: {s.mean():.6f}  Std: {s.std():.6f}  "
        f"Skew: {s.skew():.4f}  Kurt: {s.kurtosis():.4f}")
    log(f"  Min: {s.min():.4f}  Max: {s.max():.4f}")
    log(f"  Positive days: {(s>0).sum()} ({(s>0).mean()*100:.1f}%)")

log("\nFund NAV monthly returns (descriptive):")
for fund in ['akd', 'nbp', 'nti']:
    col = f'nav_return_{fund}_monthly'
    if col in monthly_agg.columns:
        s = monthly_agg[col].dropna()
        log(f"  {fund.upper()}: mean={s.mean()*100:.2f}%  "
            f"std={s.std()*100:.2f}%  min={s.min()*100:.2f}%  max={s.max()*100:.2f}%")

log("\nFund flow % of AUM (descriptive):")
for fund in ['akd', 'nbp', 'nti']:
    col = f'flow_pct_{fund}'
    if col in monthly_agg.columns:
        s = monthly_agg[col].dropna()
        log(f"  {fund.upper()}: mean={s.mean()*100:.2f}%  "
            f"std={s.std()*100:.2f}%  "
            f"inflow months={(s>0).sum()}  outflow months={(s<0).sum()}")

log()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

log("=" * 65)
log("SECTION 9 — Saving output files")
log("=" * 65)

daily.drop(columns=['month']).to_csv(
    os.path.join(BASE, "processed_data/daily_master.csv"), index=False
)
log("Saved: daily_master.csv")

monthly_agg.drop(columns=['month'], errors='ignore').to_csv(
    os.path.join(BASE, "processed_data/monthly_master.csv"), index=False
)
log("Saved: monthly_master.csv")

df_stocks.to_csv(
    os.path.join(BASE, "processed_data/kse30_stocks_daily.csv"), index=False
)
log("Saved: kse30_stocks_daily.csv")

# Write report
report_path = os.path.join(BASE, "processed_data/preprocessing_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))
log("Saved: preprocessing_report.txt")

log()
log("=" * 65)
log("PREPROCESSING COMPLETE")
log("=" * 65)
log()
log("Next steps:")
log("  1. Run EDA notebook using daily_master.csv and monthly_master.csv")
log("  2. Use monthly_master for ARIMA / LSTM fund flow prediction")
log("  3. Use daily_master for GARCH volatility modelling")
log("  4. Use kse30_stocks_daily for stock-level correlation analysis")
log("     and Markowitz weight optimisation")
