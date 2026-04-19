"""
KSE-30 INDEX FUND FLOW & REBALANCING ANALYSIS — FULL PIPELINE
==============================================================
Single file covering the complete research workflow:

  SECTION 0  — Configuration & shared utilities
  SECTION 1  — Data ingestion & cleaning  (kse30_daily_data.csv)
  SECTION 2  — Master dataset construction (daily + monthly)
  SECTION 3  — Exploratory data analysis
  SECTION 4  — GARCH volatility modelling
  SECTION 5  — Fund flow prediction  (ARIMAX + VAR)
  SECTION 6  — Market efficiency tests
  SECTION 7  — KSE-30 rebalancing weight & inclusion prediction
  SECTION 8  — Results summary

Input files (same directory as this script)
  kse30_daily_data.csv   — KSE-30 constituent daily data (2020-01-01 → present)
  funds_data.xlsx        — AKD / NBP / NTI NAV & AUM sheets
  macro_data.xlsx        — OIL / IR / USD sheets
  cpi.csv                — Monthly CPI YoY

Outputs
  daily_master.csv
  monthly_master.csv
  results_fund_flow.csv
  results_garch.csv
  results_efficiency.csv
  results_rebalancing.csv
  results_rebalancing_forecast.csv
  figures/               — all plots

Run:  python pipeline.py
Deps: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, openpyxl
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGURATION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

BASE     = os.path.dirname(os.path.abspath(__file__))
FIG_BASE = os.path.join(BASE, "figures")
for sub in ["eda","garch","fund_flow","efficiency","rebalancing","summary"]:
    os.makedirs(os.path.join(FIG_BASE, sub), exist_ok=True)

def savefig(subdir, name):
    path = os.path.join(FIG_BASE, subdir, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    fig → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
sns.set_theme(style="whitegrid", palette="muted")

FUND_COLORS  = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NIT": "#2ca02c"}
TRAIN_END    = "2023-12-31"    # fund-flow model train cut-off
ANALYSIS_START = "2021-01-04"  # align with fund NAV data start

# ── normalised company name map (SYMBOL → clean name) ────────────────────────
COMPANY_MAP = {
    "AIRLINK": "Airlink Communication Limited",
    "APL":     "Attock Petroleum Limited",
    "ATRL":    "Attock Refinery Limited",
    "AVN":     "Avanceon Limited",
    "BAFL":    "Bank Alfalah Limited",
    "BAHL":    "Bank AL-Habib Limited",
    "BOP":     "Bank of Punjab",
    "CHCC":    "Cherat Cement Company Limited",
    "CNERGY":  "Cnergyico PK Limited",
    "COLG":    "Colgate-Palmolive Pakistan Limited",
    "DAWH":    "Dawood Hercules Corporation Limited",
    "DFML":    "Dewan Farooque Motors Limited",
    "DGKC":    "D.G. Khan Cement Company Limited",
    "EFERT":   "Engro Fertilizers Limited",
    "ENGRO":   "Engro Corporation Limited",
    "ENGROH":  "Engro Holdings Limited",
    "EPCL":    "Engro Polymer & Chemicals Limited",
    "FCCL":    "Fauji Cement Company Limited",
    "FFBL":    "Fauji Fertilizer Bin Qasim Limited",
    "FFC":     "Fauji Fertilizer Company Limited",
    "GAL":     "Gul Ahmed Energy Limited",
    "GGL":     "Ghani Global Holdings Limited",
    "GHNI":    "Ghandhara Industries Limited",
    "HASCOL":  "Hascol Petroleum Limited",
    "HBL":     "Habib Bank Limited",
    "HUBC":    "The Hub Power Company Limited",
    "INIL":    "International Industries Limited",
    "ISL":     "International Steels Limited",
    "KAPCO":   "Kot Addu Power Company Limited",
    "LOTCHEM": "Lotte Chemical Pakistan Limited",
    "LUCK":    "Lucky Cement Limited",
    "MARI":    "Mari Energies Limited",
    "MCB":     "MCB Bank Limited",
    "MEBL":    "Meezan Bank Limited",
    "MLCF":    "Maple Leaf Cement Factory Limited",
    "MTL":     "Millat Tractors Limited",
    "NBP":     "National Bank of Pakistan",
    "NETSOL":  "NetSol Technologies Limited",
    "NML":     "Nishat Mills Limited",
    "NRL":     "National Refinery Limited",
    "OGDC":    "Oil & Gas Development Company Limited",
    "PAEL":    "Pak Elektron Limited",
    "PIOC":    "Pioneer Cement Limited",
    "POL":     "Pakistan Oilfields Limited",
    "PPL":     "Pakistan Petroleum Limited",
    "PRL":     "Pakistan Refinery Limited",
    "PSO":     "Pakistan State Oil Company Limited",
    "SAZEW":   "Sazgar Engineering Works Limited",
    "SEARL":   "The Searle Company Limited",
    "SHEL":    "Shell Pakistan Limited",
    "SNGP":    "Sui Northern Gas Pipelines Limited",
    "SSGC":    "Sui Southern Gas Company Limited",
    "SYS":     "Systems Limited",
    "TELE":    "Telecard Limited",
    "TPLP":    "TPL Properties Limited",
    "TREET":   "Treet Corporation Limited",
    "TRG":     "TRG Pakistan Limited",
    "UBL":     "United Bank Limited",
    "UNITY":   "Unity Foods Limited",
}

def metrics_reg(y_true, y_pred, label=""):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    da   = (np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
            if len(y_true) > 1 else np.nan)
    if label:
        print(f"    {label:28s}  RMSE={rmse:9.3f}  MAE={mae:8.3f}  "
              f"R²={r2:7.4f}  DirAcc={da:.1f}%")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": da}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA INGESTION & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 1 — Data ingestion & cleaning")
print("="*65)

# ── 1a. KSE-30 constituent data ──────────────────────────────────────────────
raw = pd.read_csv(os.path.join(BASE, "kse30_daily_data.csv"))
print(f"Raw rows: {len(raw):,}  |  columns: {list(raw.columns)}")

# Drop ISIN (not needed analytically)
raw = raw.drop(columns=["ISIN"])

# Rename columns to snake_case
raw = raw.rename(columns={
    "Date":             "date",
    "SYMBOL":           "symbol",
    "COMPANY":          "company",
    "PRICE":            "price",
    "IDX WT %":         "weight_pct",
    "FF BASED SHARES":  "ff_shares",
    "FF BASED MCAP":    "ff_mcap",
    "ORD SHARES":       "ord_shares",
    "ORD SHARES MCAP":  "ord_mcap",
    "Volume":           "vol_a",
    "VOLUME":           "vol_b",
})

raw["date"] = pd.to_datetime(raw["date"])

# Merge Volume columns: vol_a (older rows) and vol_b (newer rows) never overlap
raw["volume"] = raw["vol_a"].fillna(raw["vol_b"])
raw = raw.drop(columns=["vol_a", "vol_b"])

# Normalise company names using the canonical map
raw["company"] = raw["symbol"].map(COMPANY_MAP).fillna(raw["company"])

# Convert share/mcap columns — dash ("-") entries are trading-halt placeholder rows
for col in ["ff_shares", "ff_mcap", "ord_shares", "ord_mcap"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Drop the Oct 2021 trading-halt days (4 days, weight_pct=0, ff_shares=NaN)
# These were caused by the PSX trading suspension (Oct 25-29 2021)
halt_mask = (raw["ff_shares"].isna()) & (raw["weight_pct"] == 0)
n_halt = halt_mask.sum()
raw = raw[~halt_mask].copy()
print(f"Dropped {n_halt} trading-halt rows (Oct 2021 PSX suspension)")

# Drop any remaining zero-weight, zero-price rows (stale entries)
raw = raw[(raw["weight_pct"] > 0) | (raw["price"] > 0)].copy()

# Sort
raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

# ── 1b. Per-symbol derived features ──────────────────────────────────────────
print("Computing per-symbol features …")

raw["log_return"] = (
    raw.groupby("symbol")["price"]
    .transform(lambda x: np.log(x / x.shift(1)))
)
raw["rolling_vol_30d"] = (
    raw.groupby("symbol")["log_return"]
    .transform(lambda x: x.rolling(30, min_periods=15).std() * np.sqrt(252))
)
raw["ma_20"] = (
    raw.groupby("symbol")["price"]
    .transform(lambda x: x.rolling(20, min_periods=10).mean())
)
raw["ma_50"] = (
    raw.groupby("symbol")["price"]
    .transform(lambda x: x.rolling(50, min_periods=25).mean())
)
# Free-float market cap — used for weight reconstruction
raw["ff_mcap_computed"] = raw["ff_shares"] * raw["price"]

print(f"Stock data: {len(raw):,} rows | {raw['symbol'].nunique()} symbols | "
      f"{raw['date'].min().date()} → {raw['date'].max().date()}")

# ── 1c. Macro data ────────────────────────────────────────────────────────────
print("Loading macro data …")
macro_path = os.path.join(BASE, "macro_data.xlsx")

df_oil = pd.read_excel(macro_path, sheet_name="OIL")
df_oil = df_oil.rename(columns={"DATE":"date","PRICE":"oil_price"})
df_oil["date"] = pd.to_datetime(df_oil["date"])
df_oil["oil_log_return"] = np.log(df_oil["oil_price"] / df_oil["oil_price"].shift(1))

df_ir = pd.read_excel(macro_path, sheet_name="IR")
df_ir = df_ir.rename(columns={"DATE":"date","RATE":"interest_rate"})
df_ir["date"] = pd.to_datetime(df_ir["date"])

df_usd = pd.read_excel(macro_path, sheet_name="USD")
df_usd = df_usd.rename(columns={"DATE":"date","USD":"usdpkr"})
df_usd["date"] = pd.to_datetime(df_usd["date"])
df_usd = df_usd.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
df_usd["usdpkr_log_return"] = np.log(df_usd["usdpkr"] / df_usd["usdpkr"].shift(1))

# CPI
df_cpi = pd.read_csv(os.path.join(BASE, "cpi.csv"), skiprows=1, header=0)
df_cpi.columns = ["period_str", "cpi_yoy"]
df_cpi = df_cpi.dropna()
df_cpi["cpi_yoy"] = pd.to_numeric(df_cpi["cpi_yoy"], errors="coerce")
_month_map = {"January":"Jan","February":"Feb","March":"Mar","April":"Apr",
              "May":"May","June":"Jun","July":"Jul","August":"Aug",
              "September":"Sep","October":"Oct","November":"Nov","December":"Dec"}
def _normalise_month(s):
    for full, abbr in _month_map.items():
        if str(s).startswith(full):
            return str(s).replace(full, abbr, 1)
    return str(s)
df_cpi["period_str"] = df_cpi["period_str"].apply(_normalise_month)
df_cpi["date"] = pd.to_datetime(df_cpi["period_str"], format="%b-%y")
df_cpi["date"] = df_cpi["date"] + pd.offsets.MonthEnd(0)
df_cpi = df_cpi[["date","cpi_yoy"]].sort_values("date").reset_index(drop=True)

# ── 1d. Fund data (NAV + AUM) ─────────────────────────────────────────────────
print("Loading fund data …")
funds_path = os.path.join(BASE, "funds_data.xlsx")
funds_daily_raw   = {}
funds_monthly_raw = {}

for fund in ["AKD","NBP","NTI"]:
    df_f = pd.read_excel(funds_path, sheet_name=fund)
    df_f = df_f.rename(columns={"DATE":"date","NAV":"nav","AUM":"aum"})
    df_f["date"] = pd.to_datetime(df_f["date"])
    df_f = df_f.sort_values("date").reset_index(drop=True)
    df_f["nav_log_return"] = np.log(df_f["nav"] / df_f["nav"].shift(1))
    df_f["nav_rolling_vol_30d"] = (
        df_f["nav_log_return"].rolling(30, min_periods=15).std() * np.sqrt(252)
    )
    df_f["month"] = df_f["date"].dt.to_period("M")

    monthly_f = (
        df_f.groupby("month")
        .agg(date=("date","last"), nav_end=("nav","last"),
             nav_start=("nav","first"), aum=("aum","last"))
        .reset_index()
    )
    monthly_f["aum_prev"]     = monthly_f["aum"].shift(1)
    monthly_f["nav_return_m"] = monthly_f["nav_end"] / monthly_f["nav_start"] - 1
    monthly_f["fund_flow"]    = (
        monthly_f["aum"] - monthly_f["aum_prev"] * (1 + monthly_f["nav_return_m"])
    )
    monthly_f["fund_flow_pct"] = monthly_f["fund_flow"] / monthly_f["aum_prev"]
    flow_std  = monthly_f["fund_flow"].std()
    flow_mean = monthly_f["fund_flow"].mean()
    monthly_f["flow_spike"] = (monthly_f["fund_flow"] - flow_mean).abs() > 2*flow_std
    monthly_f = monthly_f.dropna(subset=["fund_flow"]).reset_index(drop=True)

    funds_daily_raw[fund]   = df_f
    funds_monthly_raw[fund] = monthly_f

print("Data loading complete.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — MASTER DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 2 — Master dataset construction")
print("="*65)

# Trading calendar: use dates present in the stock data
trading_days = raw["date"].drop_duplicates().sort_values().reset_index(drop=True)

# ── Forward-fill macro to daily trading calendar ──────────────────────────────
all_days = pd.date_range(trading_days.min(), trading_days.max(), freq="D")

def daily_fill(df_src, date_col, val_cols):
    base = df_src[[date_col] + val_cols].set_index(date_col)
    return (base.reindex(all_days).ffill().bfill()
            .reindex(trading_days.values).reset_index()
            .rename(columns={"index":"date"}))

macro_daily = daily_fill(df_oil, "date", ["oil_price","oil_log_return"])
macro_daily = macro_daily.merge(
    daily_fill(df_usd, "date", ["usdpkr","usdpkr_log_return"]), on="date", how="left"
)
ir_base = df_ir.set_index("date").reindex(all_days).ffill().bfill()
ir_daily = ir_base.reindex(trading_days.values).reset_index().rename(
    columns={"index":"date"}
)
macro_daily = macro_daily.merge(ir_daily, on="date", how="left")
cpi_base = df_cpi.set_index("date").reindex(all_days).ffill().bfill()
cpi_daily = cpi_base.reindex(trading_days.values).reset_index().rename(
    columns={"index":"date"}
)
macro_daily = macro_daily.merge(cpi_daily, on="date", how="left")

# ── KSE-30 index-level aggregates (from constituent data) ────────────────────
# Weight-sum sanity: fill weight_pct=0 with NaN then ffill within symbol
raw["weight_pct_clean"] = raw["weight_pct"].replace(0, np.nan)
raw["weight_pct_clean"] = raw.groupby("symbol")["weight_pct_clean"].transform(
    lambda x: x.ffill()
)
idx_daily = raw.groupby("date").agg(
    idx_total_volume  = ("volume",   "sum"),
    idx_n_stocks      = ("symbol",   "count"),
    idx_ff_mcap_total = ("ff_mcap",  "sum"),
).reset_index()
idx_daily["idx_ff_mcap_total"] = idx_daily["idx_ff_mcap_total"].replace(0, np.nan)
idx_daily["idx_log_return"] = np.log(
    idx_daily["idx_ff_mcap_total"] / idx_daily["idx_ff_mcap_total"].shift(1)
)
idx_daily["idx_rolling_vol_30d"] = (
    idx_daily["idx_log_return"].rolling(30, min_periods=15).std() * np.sqrt(252)
)

# ── Daily master ──────────────────────────────────────────────────────────────
daily = pd.DataFrame({"date": trading_days})
daily = daily.merge(idx_daily, on="date", how="left")
daily = daily.merge(macro_daily, on="date", how="left")

# Merge fund NAV data
for fund in ["AKD","NBP","NTI"]:
    f_col = fund.lower()
    df_f  = funds_daily_raw[fund]
    df_f_m = df_f[["date","nav","nav_log_return","nav_rolling_vol_30d"]].copy()
    df_f_m.columns = ["date", f"nav_{f_col}", f"nav_return_{f_col}",
                       f"nav_vol_{f_col}"]
    daily = daily.merge(df_f_m, on="date", how="left")

nav_cols = [c for c in daily.columns if c.startswith("nav_")]
daily[nav_cols] = daily[nav_cols].ffill()

# Restrict to analysis window (start with fund data)
daily = daily[daily["date"] >= ANALYSIS_START].reset_index(drop=True)
print(f"Daily master: {len(daily):,} rows × {len(daily.columns)} cols | "
      f"{daily.date.min().date()} → {daily.date.max().date()}")

null_check = daily.isnull().sum()
null_check = null_check[null_check > 0]
if len(null_check):
    print(f"Nulls: {null_check.to_dict()}")

# ── Monthly master ────────────────────────────────────────────────────────────
daily["month"] = daily["date"].dt.to_period("M")

monthly = (
    daily.groupby("month").agg(
        date                  = ("date",              "last"),
        oil_price_end         = ("oil_price",         "last"),
        oil_return_monthly    = ("oil_log_return",     "sum"),
        usdpkr_end            = ("usdpkr",             "last"),
        usdpkr_return_monthly = ("usdpkr_log_return",  "sum"),
        interest_rate_end     = ("interest_rate",      "last"),
        cpi_yoy_end           = ("cpi_yoy",            "last"),
        idx_return_monthly    = ("idx_log_return",     "sum"),
        idx_vol_monthly       = ("idx_rolling_vol_30d","last"),
    ).reset_index()
)

for fund in ["AKD","NBP","NTI"]:
    f_col = fund.lower()
    f_name_col = "nti" if fund == "NTI" else f_col
    df_m = funds_monthly_raw[fund]
    df_m["month"] = df_m["date"].dt.to_period("M")
    df_m_r = df_m[["month","nav_end","nav_return_m","aum",
                    "fund_flow","fund_flow_pct","flow_spike"]].rename(columns={
        "nav_end":       f"nav_{f_name_col}_end",
        "nav_return_m":  f"nav_return_{f_name_col}_monthly",
        "aum":           f"aum_{f_name_col}",
        "fund_flow":     f"flow_{f_name_col}",
        "fund_flow_pct": f"flow_pct_{f_name_col}",
        "flow_spike":    f"flow_spike_{f_name_col}",
    })
    monthly = monthly.merge(df_m_r, on="month", how="left")

monthly = monthly.dropna(
    subset=[c for c in monthly.columns if c.startswith("flow_") and "spike" not in c
            and "pct" not in c]
).reset_index(drop=True)

flow_cols = [c for c in monthly.columns
             if c.startswith("flow_") and not c.endswith("_pct")
             and not c.endswith("_spike")]
monthly["total_fund_flow"] = monthly[flow_cols].sum(axis=1)

# Replace inf values that arise when prior-period AUM = 0 (fund launch months)
monthly = monthly.replace([np.inf, -np.inf], np.nan)
# Drop rows where total_fund_flow is NaN (can't model them)
monthly = monthly.dropna(subset=["total_fund_flow"]).reset_index(drop=True)

print(f"Monthly master: {len(monthly):,} rows × {len(monthly.columns)} cols | "
      f"{monthly.date.min().date()} → {monthly.date.max().date()}")

# Save master datasets
daily.drop(columns=["month"]).to_csv(
    os.path.join(BASE, "daily_master.csv"), index=False
)
monthly.drop(columns=["month"], errors="ignore").to_csv(
    os.path.join(BASE, "monthly_master.csv"), index=False
)

# Save cleaned stock file
raw_out = raw.drop(columns=["weight_pct_clean"], errors="ignore")
raw_out.to_csv(os.path.join(BASE, "kse30_stocks_clean.csv"), index=False)
print("Saved: daily_master.csv, monthly_master.csv, kse30_stocks_clean.csv")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 3 — Exploratory data analysis")
print("="*65)

# ── 3.1 Descriptive stats: daily NAV returns ──────────────────────────────────
print("\n[Table] Descriptive statistics — Daily NAV log returns")
desc_rows = []
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r = daily[col].dropna().values
    jb_s, jb_p = stats.jarque_bera(r)
    desc_rows.append({
        "Fund": fund, "N": len(r),
        "Mean %": round(r.mean()*100, 4), "Std %": round(r.std()*100, 4),
        "Min %":  round(r.min()*100, 4),  "Max %": round(r.max()*100, 4),
        "Skew":   round(float(stats.skew(r)), 4),
        "ExKurt": round(float(stats.kurtosis(r)), 4),
        "JB p":   round(jb_p, 6), "Normal": "No" if jb_p < 0.05 else "Yes",
    })
desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))

# ── 3.2 AUM trend ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
for fund, col in [("AKD","aum_akd"),("NBP","aum_nbp"),("NIT","aum_nti")]:
    ax.plot(monthly["date"], monthly[col], marker="o", markersize=3,
            label=fund, color=FUND_COLORS[fund], linewidth=1.5)
ax.set_title("Monthly AUM — AKD, NBP, NIT (PKR Millions)")
ax.set_ylabel("AUM (PKR mn)"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
savefig("eda", "E01_aum_trend.png")

# ── 3.3 NAV return distributions ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Daily NAV Log Return Distributions", fontweight="bold")
for i, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                   ("NBP","nav_return_nbp"),
                                   ("NIT","nav_return_nti")]):
    s  = daily[col].dropna()
    ax = axes[i]
    ax.hist(s, bins=80, color=FUND_COLORS[fund], alpha=0.75, density=True,
            edgecolor="white")
    x  = np.linspace(s.quantile(0.01), s.quantile(0.99), 200)
    ax.plot(x, stats.norm.pdf(x, s.mean(), s.std()), "k--", linewidth=1.2)
    ax.set_title(f"{fund}  μ={s.mean():.4f}  σ={s.std():.4f}\n"
                 f"skew={s.skew():.2f}  kurt={s.kurtosis():.0f}")
    ax.set_xlim(s.quantile(0.01), s.quantile(0.99))
plt.tight_layout()
savefig("eda", "E02_nav_return_dist.png")

# ── 3.4 Fund flows bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle("Monthly Fund Flows (PKR Millions)", fontweight="bold")
for i, (fund, col, spike_col) in enumerate([
    ("AKD","flow_akd","flow_spike_akd"),
    ("NBP","flow_nbp","flow_spike_nbp"),
    ("NIT","flow_nti","flow_spike_nti"),
]):
    ax = axes[i]
    colors = [FUND_COLORS[fund] if v >= 0 else "#e74c3c"
              for v in monthly[col]]
    ax.bar(monthly["date"], monthly[col], color=colors, width=20, alpha=0.8)
    spikes = monthly[monthly[spike_col]]
    ax.scatter(spikes["date"], spikes[col], color="black", s=40, marker="*",
               zorder=5, label="Spike >2σ")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"{fund} (PKR mn)")
    ax.set_title(f"{fund} Monthly Fund Flow")
    if len(spikes): ax.legend(fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("eda", "E03_fund_flows.png")

# ── 3.5 Macro overview ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fig.suptitle("Macro Indicators", fontweight="bold")
axes[0].plot(daily["date"], daily["oil_price"], color="#c0392b", linewidth=1)
axes[0].set_ylabel("Brent (USD/bbl)")
axes[1].plot(daily["date"], daily["usdpkr"], color="#8e44ad", linewidth=1)
axes[1].set_ylabel("PKR/USD")
axes[2].step(daily["date"], daily["interest_rate"], color="#16a085",
             linewidth=1.5, where="post")
axes[2].set_ylabel("SBP Rate (%)")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("eda", "E04_macro_overview.png")

# ── 3.6 Monthly correlation heatmap ──────────────────────────────────────────
corr_cols = ["oil_return_monthly","usdpkr_return_monthly","interest_rate_end",
             "cpi_yoy_end","nav_return_akd_monthly","nav_return_nbp_monthly",
             "nav_return_nti_monthly","flow_pct_akd","flow_pct_nbp","flow_pct_nti"]
corr_labels = ["Oil ret","USD/PKR ret","Int rate","CPI YoY",
               "AKD NAV ret","NBP NAV ret","NIT NAV ret",
               "AKD flow%","NBP flow%","NIT flow%"]
corr_m = monthly[corr_cols].dropna().corr()
corr_m.index = corr_m.columns = corr_labels
fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            vmin=-1, vmax=1, mask=mask, linewidths=0.5, ax=ax,
            annot_kws={"fontsize": 8})
ax.set_title("Monthly Correlation Matrix")
plt.tight_layout()
savefig("eda", "E05_monthly_correlation.png")

# ── 3.7 KSE-30 index reconstructed return ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
idx_cum = (daily["idx_log_return"].cumsum() * 100)
ax.plot(daily["date"], idx_cum, color="#2c3e50", linewidth=1.2,
        label="KSE-30 reconstructed cumulative return")
ax.fill_between(daily["date"], idx_cum, 0,
                where=(idx_cum >= 0), alpha=0.2, color="#27ae60")
ax.fill_between(daily["date"], idx_cum, 0,
                where=(idx_cum < 0),  alpha=0.2, color="#e74c3c")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title("KSE-30 Cumulative Log Return (reconstructed from FF MCAP)")
ax.set_ylabel("Cumulative return (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
savefig("eda", "E06_index_cumulative_return.png")

# ── 3.8 Top stocks by average weight ─────────────────────────────────────────
avg_wts = (raw[raw["date"] >= ANALYSIS_START]
           .groupby("symbol")["weight_pct"].mean()
           .sort_values(ascending=False).head(15))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg_wts.index, avg_wts.values, color="#3498db", alpha=0.85)
ax.set_title("Top 15 Stocks by Average KSE-30 Weight (2021–present)")
ax.set_ylabel("Avg weight (%)")
ax.set_xticklabels(avg_wts.index, rotation=45, ha="right")
plt.tight_layout()
savefig("eda", "E07_top_weights.png")

print("EDA figures saved.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — GARCH VOLATILITY MODELLING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 4 — GARCH volatility modelling")
print("="*65)

from scipy.optimize import minimize

def _garch11_nll(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns); s2 = np.zeros(n); s2[0] = np.var(returns)
    for t in range(1, n):
        s2[t] = omega + alpha * returns[t-1]**2 + beta * s2[t-1]
    if np.any(s2 <= 0): return 1e10
    return 0.5 * np.sum(np.log(2*np.pi*s2) + returns**2/s2)

def _egarch11_nll(params, returns):
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1: return 1e10
    n = len(returns); ls2 = np.zeros(n); ls2[0] = np.log(np.var(returns))
    EZ = np.sqrt(2/np.pi)
    for t in range(1, n):
        sp = np.exp(0.5*ls2[t-1]); z = returns[t-1]/(sp+1e-10)
        ls2[t] = omega + alpha*(abs(z)-EZ) + gamma*z + beta*ls2[t-1]
    s2 = np.exp(ls2)
    if np.any(s2 <= 0) or np.any(np.isinf(s2)): return 1e10
    return 0.5 * np.sum(np.log(2*np.pi*s2) + returns**2/s2)

def fit_garch(returns, model="garch11"):
    var0 = np.var(returns); n = len(returns)
    if model == "garch11":
        res = minimize(_garch11_nll, [var0*.1,.1,.8], args=(returns,),
                       method="L-BFGS-B",
                       bounds=[(1e-8,None),(1e-6,.999),(1e-6,.999)])
        o, a, b = res.x
        s2 = np.zeros(n); s2[0] = var0
        for t in range(1,n): s2[t] = o + a*returns[t-1]**2 + b*s2[t-1]
        return {"model":"GARCH(1,1)","params":{"omega":o,"alpha":a,"beta":b},
                "sigma2":s2,"persistence":a+b,
                "aic":2*3+2*res.fun,"bic":3*np.log(n)+2*res.fun,"nll":res.fun}
    elif model == "egarch11":
        ls2_0 = np.log(var0)
        res = minimize(_egarch11_nll, [ls2_0*.1,.1,-.05,.9], args=(returns,),
                       method="L-BFGS-B",
                       bounds=[(-10,10),(-2,2),(-2,2),(-.999,.999)])
        o, a, g, b = res.x
        ls2 = np.zeros(n); ls2[0] = np.log(var0); EZ = np.sqrt(2/np.pi)
        for t in range(1,n):
            sp = np.exp(0.5*ls2[t-1]); z = returns[t-1]/(sp+1e-10)
            ls2[t] = o + a*(abs(z)-EZ) + g*z + b*ls2[t-1]
        s2 = np.exp(ls2)
        return {"model":"EGARCH(1,1)",
                "params":{"omega":o,"alpha":a,"gamma":g,"beta":b},
                "sigma2":s2,"leverage_effect":"YES" if g<0 else "NO",
                "aic":2*4+2*res.fun,"bic":4*np.log(n)+2*res.fun,"nll":res.fun}

garch_results = {}
garch_rows    = []

for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r  = daily[col].dropna().values * 100
    dt = daily.loc[daily[col].notna(), "date"].values
    print(f"\n  {fund} — fitting GARCH(1,1) and EGARCH(1,1) …")

    g11  = fit_garch(r, "garch11")
    eg11 = fit_garch(r, "egarch11")

    best = g11 if g11["aic"] <= eg11["aic"] else eg11
    print(f"    GARCH(1,1):  AIC={g11['aic']:.1f}  "
          f"persist={g11['persistence']:.4f}")
    print(f"    EGARCH(1,1): AIC={eg11['aic']:.1f}  "
          f"leverage={eg11['leverage_effect']}")
    print(f"    Best: {best['model']}")

    garch_results[fund] = {"g11":g11,"eg11":eg11,"best":best,"returns":r,"dates":dt}
    garch_rows.append({
        "Fund": fund, "Model": "GARCH(1,1)",
        "omega": round(g11["params"]["omega"],6),
        "alpha": round(g11["params"]["alpha"],4),
        "beta":  round(g11["params"]["beta"],4),
        "persist": round(g11["persistence"],4),
        "AIC":   round(g11["aic"],2),
    })

pd.DataFrame(garch_rows).to_csv(
    os.path.join(BASE, "results_garch.csv"), index=False
)
print("\n  Saved: results_garch.csv")

# ── GARCH figures ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle("Daily NAV Returns & GARCH(1,1) Conditional Volatility",
             fontweight="bold")
for row, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                    ("NBP","nav_return_nbp"),
                                    ("NIT","nav_return_nti")]):
    r  = daily[col].dropna().values * 100
    dt = daily.loc[daily[col].notna(), "date"].values
    vol = np.sqrt(garch_results[fund]["g11"]["sigma2"])
    axes[row][0].fill_between(dt, r, alpha=0.5, color=FUND_COLORS[fund])
    axes[row][0].axhline(0, color="black", linewidth=0.6)
    axes[row][0].set_title(f"{fund} — Daily NAV Returns (%)")
    axes[row][0].set_ylabel("Return (%)")
    axes[row][1].plot(dt, vol, color=FUND_COLORS[fund], linewidth=1)
    axes[row][1].set_title(f"{fund} — GARCH(1,1) Cond. Vol (%)")
    axes[row][1].set_ylabel("Cond. Std Dev (%)")
    for ax in axes[row]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("garch", "G01_returns_and_vol.png")

# VaR backtest
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle("GARCH(1,1) 5% VaR Backtest", fontweight="bold")
for i, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                   ("NBP","nav_return_nbp"),
                                   ("NIT","nav_return_nti")]):
    r   = daily[col].dropna().values * 100
    dt  = daily.loc[daily[col].notna(), "date"].values
    var_line = stats.norm.ppf(0.05) * np.sqrt(garch_results[fund]["g11"]["sigma2"])
    exceed   = r < var_line
    axes[i].fill_between(dt, r, 0, where=(r>=0), alpha=0.4,
                          color=FUND_COLORS[fund])
    axes[i].fill_between(dt, r, 0, where=(r<0), alpha=0.4, color="#e74c3c")
    axes[i].plot(dt, var_line, color="black", linewidth=1, label="5% VaR")
    axes[i].scatter(dt[exceed], r[exceed], color="black", s=8, zorder=5,
                    label=f"Exc. ({exceed.sum()}, {exceed.mean()*100:.1f}%)")
    axes[i].set_title(f"{fund}"); axes[i].legend(fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("garch", "G02_var_backtest.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FUND FLOW PREDICTION  (ARIMAX + VAR)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 5 — Fund flow prediction")
print("="*65)

MACRO_COLS = ["interest_rate_end","cpi_yoy_end",
              "oil_return_monthly","usdpkr_return_monthly"]
TARGET     = "total_fund_flow"

train_m = monthly["date"] <= TRAIN_END
test_m  = ~train_m
print(f"Train: {train_m.sum()} months | Test: {test_m.sum()} months")

# ── 5.1 Stationarity (ADF) ───────────────────────────────────────────────────
print("\n  ADF tests:")
def adf_simple(series, name):
    s = series.dropna().replace([np.inf,-np.inf], np.nan).dropna().values.astype(float)
    if len(s) < 5:
        print(f"    {name:30s}  insufficient data"); return 1.0
    dy = np.diff(s); yl = s[:-1]
    X = np.column_stack([np.ones(len(yl)), yl])
    beta, _, _, _ = lstsq(X, dy)
    resid = dy - X @ beta
    s2 = resid@resid/(len(dy)-2)
    vb = s2 * np.linalg.inv(X.T@X)[1,1]
    t  = beta[1]/np.sqrt(max(vb,1e-15))
    p  = stats.t.sf(abs(t), df=len(dy)-2)*2
    print(f"    {name:30s}  t={t:7.3f}  p={p:.4f}  "
          f"{'Stationary' if p<0.05 else 'Non-stationary'}")
    return p

for col in [TARGET]+MACRO_COLS:
    adf_simple(monthly[col], col)

# ── 5.2 Granger causality ────────────────────────────────────────────────────
print("\n  Granger causality (macro → fund flow):")
gc_rows = []
def granger_test(y, x, lags=3, label=""):
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)
    results = []
    for lag in range(1, lags+1):
        df_g = pd.DataFrame({"y":y,"x":x})
        for L in range(1,lag+1):
            df_g[f"yl{L}"] = df_g["y"].shift(L)
            df_g[f"xl{L}"] = df_g["x"].shift(L)
        df_g = df_g.dropna().replace([np.inf,-np.inf], np.nan).dropna()
        yv = df_g["y"].values.astype(float); n = len(yv)
        Xr = np.column_stack([np.ones(n)]+[df_g[f"yl{L}"].values.astype(float)
                               for L in range(1,lag+1)])
        br,_,_,_ = lstsq(Xr,yv); rss_r = np.sum((yv-Xr@br)**2)
        Xu = np.column_stack([Xr]+[df_g[f"xl{L}"].values.astype(float)
                               for L in range(1,lag+1)])
        bu,_,_,_ = lstsq(Xu,yv); rss_u = np.sum((yv-Xu@bu)**2)
        kn,ku = Xr.shape[1],Xu.shape[1]; dd = n-ku
        F = ((rss_r-rss_u)/(ku-kn))/(rss_u/dd) if dd>0 else np.nan
        p = 1-stats.f.cdf(F,ku-kn,dd) if not np.isnan(F) else np.nan
        results.append((lag,F,p))
    return results

y_flow = monthly[TARGET].values
for macro_col, label in [("interest_rate_end","IR → flow"),
                          ("cpi_yoy_end","CPI → flow"),
                          ("oil_return_monthly","Oil → flow"),
                          ("usdpkr_return_monthly","USD/PKR → flow")]:
    res = granger_test(y_flow, monthly[macro_col].values, lags=3, label=label)
    for lag, F, p in res[:1]:
        sig = "**" if p<0.05 else ("*" if p<0.10 else "ns")
        print(f"    {label:20s} lag={lag}: F={F:.3f}  p={p:.4f}  {sig}")
        gc_rows.append({"Variable":label,"Lag":lag,"F":round(F,3),"p":round(p,4),"Sig":sig})

# ── 5.3 ARIMAX model ─────────────────────────────────────────────────────────
print("\n  ARIMAX(1,0,1) …")
df_m = monthly[[TARGET]+MACRO_COLS].dropna()
tr_df = df_m[monthly["date"][df_m.index] <= TRAIN_END]
te_df = df_m[monthly["date"][df_m.index] >  TRAIN_END]
y_tr  = tr_df[TARGET].values.astype(float)
y_te  = te_df[TARGET].values.astype(float)
X_tr  = tr_df[MACRO_COLS].values.astype(float)
X_te  = te_df[MACRO_COLS].values.astype(float)

def fit_arimax(y_train, X_train, y_test, X_test, p=1):
    n_tr = len(y_train)
    rows = []
    for t in range(p, n_tr):
        row = [y_train[t-i] for i in range(1,p+1)] + list(X_train[t])
        rows.append(row)
    Xd = np.column_stack([np.ones(len(rows))]+
                          [np.array(rows)[:,i]
                           for i in range(np.array(rows).shape[1])])
    yd = y_train[p:]
    beta,_,_,_ = lstsq(Xd,yd)
    fitted = Xd@beta
    r2_tr  = 1 - np.var(yd-fitted)/np.var(yd)
    history = list(y_train); preds = []
    for t in range(len(y_test)):
        ar_l = [history[-(i)] for i in range(1,p+1)]
        x_n  = np.array([1.0]+ar_l+list(X_test[t]))
        preds.append(x_n@beta); history.append(y_test[t])
    return np.array(preds), fitted, beta, r2_tr

arimax_pred, arimax_fit, arimax_beta, arimax_r2_tr = fit_arimax(
    y_tr, X_tr, y_te, X_te, p=1)
m_arimax = metrics_reg(y_te, arimax_pred, "ARIMAX(1,0,1)")
m_naive  = metrics_reg(y_te, [y_tr[-1]]+list(y_te[:-1]), "Naive (RW)     ")

# ── 5.4 VAR(1) model ─────────────────────────────────────────────────────────
print("  VAR(1) …")
ENDO = ["total_fund_flow","interest_rate_end","cpi_yoy_end"]
EXOG = ["oil_return_monthly","usdpkr_return_monthly"]
var_tr = monthly[monthly["date"]<=TRAIN_END][ENDO+EXOG].dropna()
var_te = monthly[monthly["date"]>TRAIN_END][ENDO+EXOG].dropna()
Y_tr = var_tr[ENDO].values.astype(float)
X_tr_v = var_tr[EXOG].values.astype(float)
Y_te = var_te[ENDO].values.astype(float)
X_te_v = var_te[EXOG].values.astype(float)
Z_tr = np.column_stack([np.ones(len(Y_tr)-1), Y_tr[:-1], X_tr_v[1:]])
betas_var = [lstsq(Z_tr, Y_tr[1:,i])[0] for i in range(len(ENDO))]

history_Y = list(Y_tr); var_preds = []
for t in range(len(Y_te)):
    z = np.concatenate([[1.0], np.array(history_Y[-1]), X_te_v[t]])
    var_preds.append(z @ betas_var[0]); history_Y.append(Y_te[t])
m_var = metrics_reg(y_te, np.array(var_preds), "VAR(1)         ")

# ── 5.5 Individual fund ARIMAX ────────────────────────────────────────────────
fund_ff_results = {}
for fund, col in [("AKD","flow_akd"),("NBP","flow_nbp"),("NIT","flow_nti")]:
    df_f = monthly[[col]+MACRO_COLS].dropna()
    tr_f = df_f[monthly["date"][df_f.index]<=TRAIN_END]
    te_f = df_f[monthly["date"][df_f.index]>TRAIN_END]
    if len(te_f)==0: continue
    pred_f,_,_,_ = fit_arimax(
        tr_f[col].values.astype(float), tr_f[MACRO_COLS].values.astype(float),
        te_f[col].values.astype(float), te_f[MACRO_COLS].values.astype(float), p=1)
    m_f = metrics_reg(te_f[col].values, pred_f, f"{fund} ARIMAX    ")
    fund_ff_results[fund] = {"pred":pred_f,"actual":te_f[col].values,"metrics":m_f}

# ── 5.6 Fund flow figures ─────────────────────────────────────────────────────
dates_tr = monthly.loc[train_m,"date"].values
dates_te = monthly.loc[test_m, "date"].values

fig, ax = plt.subplots(figsize=(13,5))
ax.bar(dates_tr, monthly.loc[train_m,TARGET],
       color=["#2980b9" if v>=0 else "#e74c3c"
              for v in monthly.loc[train_m,TARGET]], width=20, alpha=0.5,
       label="Actual (train)")
ax.bar(dates_te, y_te,
       color=["#2980b9" if v>=0 else "#e74c3c" for v in y_te],
       width=20, alpha=0.85, label="Actual (test)")
ax.plot(dates_te, arimax_pred, "o-", color="#e74c3c", linewidth=2,
        markersize=5, label=f"ARIMAX (R²={m_arimax['R2']:.3f})")
ax.plot(dates_te, np.array(var_preds), "s--", color="#2ca02c", linewidth=2,
        markersize=5, label=f"VAR(1) (R²={m_var['R2']:.3f})")
ax.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Total Fund Flow — Actual vs ARIMAX vs VAR(1) (PKR Millions)")
ax.set_ylabel("Flow (PKR mn)"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
savefig("fund_flow", "FF01_total_flow_predictions.png")

# Granger p-value bar chart
fig, ax = plt.subplots(figsize=(9,4))
gc_df = pd.DataFrame(gc_rows)
cols_bar = ["#e74c3c" if p<0.05 else "#f39c12" if p<0.10 else "#95a5a6"
            for p in gc_df["p"]]
ax.bar(gc_df["Variable"], gc_df["p"], color=cols_bar, alpha=0.85)
ax.axhline(0.05, color="black", linewidth=1.2, linestyle="--", label="5%")
ax.axhline(0.10, color="gray",  linewidth=0.8, linestyle=":",  label="10%")
ax.set_ylabel("Granger p-value (lag 1)")
ax.set_title("Granger Causality: Does Macro Predict Fund Flow?")
ax.legend(); ax.set_xticklabels(gc_df["Variable"], rotation=20, ha="right")
plt.tight_layout()
savefig("fund_flow", "FF02_granger.png")

# Save results
ff_rows = [
    {"Model":"Naive (RW)","Target":"Total","RMSE":round(m_naive["RMSE"],2),
     "MAE":round(m_naive["MAE"],2),"R2":round(m_naive["R2"],4),
     "DirAcc":round(m_naive["DirAcc"],1),"Note":"Benchmark"},
    {"Model":"ARIMAX(1,0,1)","Target":"Total","RMSE":round(m_arimax["RMSE"],2),
     "MAE":round(m_arimax["MAE"],2),"R2":round(m_arimax["R2"],4),
     "DirAcc":round(m_arimax["DirAcc"],1),"Note":"Primary"},
    {"Model":"VAR(1)","Target":"Total","RMSE":round(m_var["RMSE"],2),
     "MAE":round(m_var["MAE"],2),"R2":round(m_var["R2"],4),
     "DirAcc":round(m_var["DirAcc"],1),"Note":"System model"},
]
for fund in ["AKD","NBP","NIT"]:
    if fund in fund_ff_results:
        m = fund_ff_results[fund]["metrics"]
        ff_rows.append({
            "Model":"ARIMAX(1,0,1)","Target":fund,
            "RMSE":round(m["RMSE"],2),"MAE":round(m["MAE"],2),
            "R2":round(m["R2"],4),"DirAcc":round(m["DirAcc"],1),
            "Note":"Heterogeneity"})
pd.DataFrame(ff_rows).to_csv(
    os.path.join(BASE,"results_fund_flow.csv"), index=False)
print("  Saved: results_fund_flow.csv")

print(f"\n  Model comparison:")
print(f"  {'Model':20s}  {'RMSE':>10}  {'R²':>8}  {'DirAcc':>8}")
for r in ff_rows[:3]:
    print(f"  {r['Model']:20s}  {r['RMSE']:>10.1f}  {r['R2']:>8.4f}  "
          f"{r['DirAcc']:>8.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — MARKET EFFICIENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 6 — Market efficiency tests")
print("="*65)

def runs_test(r, name=""):
    r = r[~np.isnan(r)]; signs = np.where(r>=0,1,-1)
    np_, nn = (signs==1).sum(),(signs==-1).sum(); n = np_+nn
    n_runs = 1+(signs[1:]!=signs[:-1]).sum()
    exp = 2*np_*nn/n+1
    var = 2*np_*nn*(2*np_*nn-n)/(n**2*(n-1))
    Z = (n_runs-exp)/np.sqrt(var) if var>0 else np.nan
    p = 2*(1-stats.norm.cdf(abs(Z))) if not np.isnan(Z) else np.nan
    return {"name":name,"Z":round(Z,4),"p":round(p,4),
            "verdict":"Inefficient" if p<0.05 else "Efficient"}

def vr_test(r, q=2, name=""):
    r = r[~np.isnan(r)]; T = len(r)
    rq = [np.sum(r[i:i+q]) for i in range(T-q+1)]
    s1 = np.var(r,ddof=1); sq = np.var(rq,ddof=1)/q
    VR = sq/s1 if s1>0 else np.nan
    mu = r.mean(); numer = np.sum((r-mu)**2)**2
    delta = np.array([np.sum((r[k:]-mu)**2*(r[:-k]-mu)**2)/numer*T
                       for k in range(1,q)])
    theta = 4*sum((1-k/q)**2*delta[k-1] for k in range(1,q))
    Zs = (VR-1)/np.sqrt(theta/T) if theta>0 else np.nan
    p  = 2*(1-stats.norm.cdf(abs(Zs))) if not np.isnan(Zs) else np.nan
    return {"name":name,"q":q,"VR":round(VR,4),"Z":round(Zs,4),
            "p":round(p,4),"verdict":"Inefficient" if p<0.05 else "Efficient"}

def lb_test(r, lags=10, name=""):
    r = r[~np.isnan(r)]; n = len(r); mu = r.mean()
    acf = [np.sum((r[k:]-mu)*(r[:-k]-mu))/np.sum((r-mu)**2)
           for k in range(1,lags+1)]
    Q = n*(n+2)*sum(rk**2/(n-k) for k,rk in enumerate(acf,1))
    p = 1-stats.chi2.cdf(Q,df=lags)
    return {"name":name,"Q":round(Q,4),"p":round(p,4),"acf1":round(acf[0],4),
            "verdict":"Inefficient" if p<0.05 else "Efficient"}

def hurst_exp(r, name=""):
    r = r[~np.isnan(r)]; n = len(r)
    lags = range(2,min(n//4,80))
    RS = []
    for lag in lags:
        subs = [r[i:i+lag] for i in range(0,n-lag+1,lag)]
        sub_rs = []
        for sub in subs:
            if len(sub)<2: continue
            d = np.cumsum(sub-sub.mean()); s = np.std(sub,ddof=1)
            if s>0: sub_rs.append((d.max()-d.min())/s)
        if sub_rs: RS.append(np.mean(sub_rs))
    valid = [(lag,rs) for lag,rs in zip(lags,RS) if rs>0]
    if len(valid)<5: return None
    H = stats.linregress(np.log([v[0] for v in valid]),
                          np.log([v[1] for v in valid]))[0]
    interp = ("Persistent" if H>0.55 else
               "Mean-reverting" if H<0.45 else "Random walk")
    return {"name":name,"H":round(H,4),"interpretation":interp}

eff_rows = []
print(f"\n  {'Fund':6s} {'Runs Z':>8} {'Runs p':>8} {'VR(2)':>7} "
      f"{'VR p':>7} {'LB p':>7} {'Hurst H':>9}")
print("  " + "-"*60)

for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r = daily[col].dropna().values
    rt  = runs_test(r, fund)
    vr2 = vr_test(r, q=2, name=fund)
    vr4 = vr_test(r, q=4, name=fund)
    lb  = lb_test(r, lags=10, name=fund)
    H   = hurst_exp(r, fund)
    print(f"  {fund:6s} {rt['Z']:>8.4f} {rt['p']:>8.4f} "
          f"{vr2['VR']:>7.4f} {vr2['p']:>7.4f} {lb['p']:>7.4f} "
          f"{H['H'] if H else 'N/A':>9}")
    eff_rows.append({
        "Fund":fund, "N":len(r),
        "Runs Z":rt["Z"],"Runs p":rt["p"],"Runs verdict":rt["verdict"],
        "VR(2)":vr2["VR"],"VR(2) p":vr2["p"],"VR(4)":vr4["VR"],
        "VR verdict":vr2["verdict"],
        "LB Q p":lb["p"],"ACF lag-1":lb["acf1"],
        "Hurst H": H["H"] if H else np.nan,
        "Hurst interp": H["interpretation"] if H else "",
    })

pd.DataFrame(eff_rows).to_csv(
    os.path.join(BASE,"results_efficiency.csv"), index=False)

# Sub-period analysis
print("\n  Sub-period (high-rate Apr 2022–Dec 2023 vs low-rate):")
high = (daily["date"]>="2022-04-01")&(daily["date"]<="2023-12-31")
low  = ~high
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    rh = daily.loc[high,col].dropna().values
    rl = daily.loc[low, col].dropna().values
    vr_h = vr_test(rh,q=2); vr_l = vr_test(rl,q=2)
    Hh = hurst_exp(rh); Hl = hurst_exp(rl)
    print(f"  {fund}: VR(2) high-rate={vr_h['VR']:.4f} "
          f"low-rate={vr_l['VR']:.4f} | "
          f"H high={Hh['H'] if Hh else 'N/A':.4f} "
          f"low={Hl['H'] if Hl else 'N/A':.4f}")

# Efficiency figures
fig, axes = plt.subplots(1,3,figsize=(14,4))
fig.suptitle("ACF — Daily NAV Returns", fontweight="bold")
for i,(fund,col) in enumerate([("AKD","nav_return_akd"),
                                 ("NBP","nav_return_nbp"),
                                 ("NIT","nav_return_nti")]):
    r  = daily[col].dropna().values; n = len(r)
    ci = 1.96/np.sqrt(n)
    acf_v = [np.corrcoef(r[k:],r[:-k])[0,1] for k in range(1,21)]
    colors = ["#e74c3c" if abs(v)>ci else FUND_COLORS[fund] for v in acf_v]
    axes[i].bar(range(1,21),acf_v,color=colors,alpha=0.8)
    axes[i].axhline(ci,color="black",linewidth=0.8,linestyle="--")
    axes[i].axhline(-ci,color="black",linewidth=0.8,linestyle="--")
    axes[i].axhline(0,color="black",linewidth=0.4)
    row = next(r for r in eff_rows if r["Fund"]==fund)
    axes[i].set_title(f"{fund}  LB p={row['LB Q p']:.4f}\n"
                      f"ACF₁={row['ACF lag-1']:.4f}")
    axes[i].set_xlabel("Lag")
plt.tight_layout()
savefig("efficiency","EF01_acf.png")

fig, ax = plt.subplots(figsize=(10,5))
for fund,col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r = daily[col].dropna().values
    vrs = [vr_test(r,q)["VR"] for q in [2,4,8,16]]
    ax.plot([2,4,8,16],vrs,marker="o",color=FUND_COLORS[fund],
            linewidth=1.5,label=fund)
ax.axhline(1.0,color="black",linewidth=1.2,linestyle="--",label="VR=1 (RW)")
ax.set_xlabel("q"); ax.set_ylabel("VR(q)")
ax.set_title("Variance Ratio Test across Holding Periods")
ax.legend(); ax.set_xticks([2,4,8,16])
plt.tight_layout()
savefig("efficiency","EF02_variance_ratio.png")

print("  Saved: results_efficiency.csv")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — REBALANCING WEIGHT & INCLUSION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 7 — Rebalancing weight & inclusion prediction")
print("="*65)

# ── 7.1 Detect rebalancing dates from composition changes ─────────────────────
stocks_clean = raw.copy()
dates_all    = sorted(stocks_clean["date"].unique())

# Use full date range (not restricted to ANALYSIS_START for better history)
REBAL_DATES = []
prev_syms   = None
for d in dates_all:
    curr_syms = set(stocks_clean[stocks_clean["date"]==d]["symbol"])
    if prev_syms is not None and curr_syms != prev_syms:
        REBAL_DATES.append(d)
    prev_syms = curr_syms
REBAL_DATES = pd.DatetimeIndex(REBAL_DATES)
print(f"  Detected {len(REBAL_DATES)} rebalancing dates:")
for d in REBAL_DATES:
    print(f"    {d.date()}")

def nearest_date(target, available, window=5):
    cands = [d for d in available
             if abs((pd.Timestamp(d)-pd.Timestamp(target)).days)<=window]
    return min(cands, key=lambda d: abs(
        (pd.Timestamp(d)-pd.Timestamp(target)).days)) if cands else None

# ── 7.2 Build panel ───────────────────────────────────────────────────────────
print("\n  Building rebalancing panel …")
FEAT_COLS = ["cur_weight","mom_30","mom_60","mom_90","vol_30","vol_60",
             "avg_volume","mkt_cap_proxy","price_to_ma20","price_to_ma50",
             "wt_drift","wt_range"]
records = []

for wi, rdate in enumerate(REBAL_DATES[:-1]):
    next_rdate   = REBAL_DATES[wi+1]
    rd_actual    = nearest_date(rdate, dates_all)
    next_actual  = nearest_date(next_rdate, dates_all)
    if rd_actual is None or next_actual is None: continue

    snap_end = pd.Timestamp(rd_actual) - pd.Timedelta(days=1)
    curr_wts = stocks_clean[stocks_clean["date"]==pd.Timestamp(rd_actual)].set_index(
        "symbol")["weight_pct"]
    next_wts = stocks_clean[stocks_clean["date"]==pd.Timestamp(next_actual)].set_index(
        "symbol")["weight_pct"]
    curr_syms = set(curr_wts.index)
    next_syms = set(next_wts.index)

    for sym in curr_syms:
        s30 = stocks_clean[(stocks_clean["date"]>=snap_end-pd.Timedelta(days=30))&
                            (stocks_clean["date"]<=snap_end)&
                            (stocks_clean["symbol"]==sym)]
        s60 = stocks_clean[(stocks_clean["date"]>=snap_end-pd.Timedelta(days=60))&
                            (stocks_clean["date"]<=snap_end)&
                            (stocks_clean["symbol"]==sym)]
        s90 = stocks_clean[(stocks_clean["date"]>=snap_end-pd.Timedelta(days=90))&
                            (stocks_clean["date"]<=snap_end)&
                            (stocks_clean["symbol"]==sym)]
        if len(s30)<5: continue
        lp = s30["price"].iloc[-1]
        records.append({
            "win_idx": wi, "rebal_date": pd.Timestamp(rd_actual),
            "next_rebal_date": pd.Timestamp(next_actual), "symbol": sym,
            "cur_weight":    curr_wts.get(sym, np.nan),
            "mom_30":        s30["log_return"].sum(),
            "mom_60":        s60["log_return"].sum() if len(s60)>=10 else s30["log_return"].sum(),
            "mom_90":        s90["log_return"].sum() if len(s90)>=15 else s30["log_return"].sum(),
            "vol_30":        s30["log_return"].std()*np.sqrt(252),
            "vol_60":        s60["log_return"].std()*np.sqrt(252) if len(s60)>=10
                             else s30["log_return"].std()*np.sqrt(252),
            "avg_volume":    s30["volume"].mean(),
            "mkt_cap_proxy": lp * s30["volume"].mean(),
            "price_to_ma20": (lp/s30["ma_20"].iloc[-1]-1)
                             if s30["ma_20"].iloc[-1]>0 else 0,
            "price_to_ma50": (lp/s30["ma_50"].iloc[-1]-1)
                             if s30["ma_50"].iloc[-1]>0 else 0,
            "wt_drift":      s30["weight_pct"].iloc[-1]-s30["weight_pct"].iloc[0]
                             if len(s30)>1 else 0,
            "wt_range":      s30["weight_pct"].max()-s30["weight_pct"].min()
                             if len(s30)>1 else 0,
            "target_weight": next_wts.get(sym, 0.0),
            "retained":      1 if sym in next_syms else 0,
        })

panel = pd.DataFrame(records)
print(f"  Panel: {len(panel)} rows | {panel.win_idx.nunique()} windows | "
      f"{panel.symbol.nunique()} symbols")
print(f"  Retained rate: {panel.retained.mean()*100:.1f}%")

# Fill NaN with train medians
n_wins = panel.win_idx.nunique()
TRAIN_WINS = list(range(max(1, n_wins-2)))   # all but last 2 windows
TEST_WINS  = list(range(max(1,n_wins-2), n_wins))
train_meds = panel[panel.win_idx.isin(TRAIN_WINS)][FEAT_COLS].median()
panel[FEAT_COLS] = panel[FEAT_COLS].fillna(train_meds)

# ── 7.3 Feature stats: retained vs dropped ───────────────────────────────────
print("\n  Feature comparison — retained vs dropped:")
print(f"  {'Feature':20s} {'Retained':>10} {'Dropped':>10} {'p-value':>10}")
print("  " + "-"*52)
for feat in FEAT_COLS:
    rv = panel[panel.retained==1][feat].dropna()
    dv = panel[panel.retained==0][feat].dropna()
    if len(dv)<3: continue
    _, p = stats.mannwhitneyu(rv, dv, alternative="two-sided")
    sig = "**" if p<0.05 else ("*" if p<0.10 else "")
    print(f"  {feat:20s} {rv.mean():>10.4f} {dv.mean():>10.4f} {p:>10.4f} {sig}")

# ── 7.4 Train/test split & models ────────────────────────────────────────────
train_p = panel[panel.win_idx.isin(TRAIN_WINS)].copy()
test_p  = panel[panel.win_idx.isin(TEST_WINS)].copy()
print(f"\n  Train: {len(train_p)} | Test: {len(test_p)}")

scaler = StandardScaler()
X_tr_p = scaler.fit_transform(train_p[FEAT_COLS].values)
X_te_p = scaler.transform(test_p[FEAT_COLS].values)
y_wt_tr = train_p["target_weight"].values
y_wt_te = test_p["target_weight"].values
y_ret_tr = train_p["retained"].values
y_ret_te = test_p["retained"].values

# Task A: weight regression
ridge = Ridge(alpha=1.0).fit(X_tr_p, y_wt_tr)
rf_r  = RandomForestRegressor(n_estimators=100, max_depth=4,
                               min_samples_leaf=3, random_state=42
                               ).fit(X_tr_p, y_wt_tr)
m_naive_wt = metrics_reg(y_wt_te, test_p["cur_weight"].values, "Naive wt      ")
m_ridge_wt = metrics_reg(y_wt_te, ridge.predict(X_te_p),       "Ridge         ")
m_rf_wt    = metrics_reg(y_wt_te, rf_r.predict(X_te_p),        "Random Forest ")

# Task B: inclusion classification
logit = LogisticRegression(C=0.5, max_iter=1000,
                            random_state=42).fit(X_tr_p, y_ret_tr)
rf_c  = RandomForestClassifier(n_estimators=100, max_depth=3,
                                min_samples_leaf=3,
                                random_state=42).fit(X_tr_p, y_ret_tr)

def clf_m(yt, yp, yprob=None, label=""):
    acc = accuracy_score(yt,yp)
    auc = roc_auc_score(yt,yprob) if yprob is not None else np.nan
    tp = int(((yt==1)&(yp==1)).sum())
    fp = int(((yt==0)&(yp==1)).sum())
    fn = int(((yt==1)&(yp==0)).sum())
    tn = int(((yt==0)&(yp==0)).sum())
    if label:
        print(f"    {label:28s}  Acc={acc:.3f}  AUC={auc:.3f}  "
              f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"Accuracy":acc,"AUC":auc,"TP":tp,"FP":fp,"FN":fn,"TN":tn}

m_naive_clf = clf_m(y_ret_te, np.ones(len(y_ret_te),int), label="Naive (all ret) ")
m_logit     = clf_m(y_ret_te, logit.predict(X_te_p),
                    logit.predict_proba(X_te_p)[:,1], "Logistic        ")
m_rf_clf    = clf_m(y_ret_te, rf_c.predict(X_te_p),
                    rf_c.predict_proba(X_te_p)[:,1],  "Random Forest   ")

# ── 7.5 Forward prediction: next rebalancing ──────────────────────────────────
print("\n  Predicting next rebalancing …")
last_rebal  = REBAL_DATES[-1]
pred_target = last_rebal + pd.DateOffset(months=6)  # ~6 months ahead
snap_end_f  = last_rebal - pd.Timedelta(days=1)

curr_syms_f = set(stocks_clean[stocks_clean["date"]==
                  pd.Timestamp(nearest_date(last_rebal,dates_all))]["symbol"])
curr_wts_f  = stocks_clean[stocks_clean["date"]==
              pd.Timestamp(nearest_date(last_rebal,dates_all))].set_index(
              "symbol")["weight_pct"]

future_feats = []
for sym in curr_syms_f:
    s30 = stocks_clean[(stocks_clean["date"]>=snap_end_f-pd.Timedelta(days=30))&
                        (stocks_clean["date"]<=snap_end_f)&
                        (stocks_clean["symbol"]==sym)]
    if len(s30)<5: continue
    s60 = stocks_clean[(stocks_clean["date"]>=snap_end_f-pd.Timedelta(days=60))&
                        (stocks_clean["date"]<=snap_end_f)&
                        (stocks_clean["symbol"]==sym)]
    s90 = stocks_clean[(stocks_clean["date"]>=snap_end_f-pd.Timedelta(days=90))&
                        (stocks_clean["date"]<=snap_end_f)&
                        (stocks_clean["symbol"]==sym)]
    lp = s30["price"].iloc[-1]
    cw = curr_wts_f.get(sym, np.nan)
    future_feats.append({
        "symbol": sym, "cur_weight": cw,
        "mom_30": s30["log_return"].sum(),
        "mom_60": s60["log_return"].sum() if len(s60)>=10 else s30["log_return"].sum(),
        "mom_90": s90["log_return"].sum() if len(s90)>=15 else s30["log_return"].sum(),
        "vol_30": s30["log_return"].std()*np.sqrt(252),
        "vol_60": s60["log_return"].std()*np.sqrt(252) if len(s60)>=10
                  else s30["log_return"].std()*np.sqrt(252),
        "avg_volume":    s30["volume"].mean(),
        "mkt_cap_proxy": lp*s30["volume"].mean(),
        "price_to_ma20": (lp/s30["ma_20"].iloc[-1]-1)
                         if "ma_20" in s30.columns and s30["ma_20"].iloc[-1]>0 else 0,
        "price_to_ma50": (lp/s30["ma_50"].iloc[-1]-1)
                         if "ma_50" in s30.columns and s30["ma_50"].iloc[-1]>0 else 0,
        "wt_drift": s30["weight_pct"].iloc[-1]-s30["weight_pct"].iloc[0]
                    if len(s30)>1 else 0,
        "wt_range": s30["weight_pct"].max()-s30["weight_pct"].min()
                    if len(s30)>1 else 0,
    })

future_df = pd.DataFrame(future_feats).fillna(train_meds)
X_fut = scaler.transform(future_df[FEAT_COLS].values)

future_df["pred_wt_ridge"]     = ridge.predict(X_fut)
future_df["pred_wt_rf"]        = rf_r.predict(X_fut)
future_df["pred_wt_avg"]       = (future_df["pred_wt_ridge"]+
                                   future_df["pred_wt_rf"])/2
future_df["ret_prob_logit"]    = logit.predict_proba(X_fut)[:,1]
future_df["ret_prob_rf"]       = rf_c.predict_proba(X_fut)[:,1]
future_df["avg_ret_prob"]      = (future_df["ret_prob_logit"]+
                                   future_df["ret_prob_rf"])/2
future_df["exclusion_risk"]    = 1 - future_df["avg_ret_prob"]
future_df = future_df.sort_values("exclusion_risk", ascending=False)

print(f"\n  Forward forecast (next rebalancing ~{pred_target.date()}):")
print(f"  {'Symbol':8s} {'CurWt':>7} {'PredWt':>8} {'RetProb':>9} {'Risk':>8}")
print("  " + "-"*45)
for _, row in future_df.iterrows():
    flag = " ⚠" if row["exclusion_risk"]>0.35 else ""
    print(f"  {row['symbol']:8s} {row['cur_weight']:>7.2f} "
          f"{row['pred_wt_avg']:>8.2f} {row['avg_ret_prob']:>9.3f} "
          f"{row['exclusion_risk']:>8.3f}{flag}")

# ── 7.6 Rebalancing figures ───────────────────────────────────────────────────
# Retention probability chart
fig, ax = plt.subplots(figsize=(12,6))
fsort = future_df.sort_values("avg_ret_prob")
colors = ["#e74c3c" if p<0.65 else "#f39c12" if p<0.80 else "#2ecc71"
          for p in fsort["avg_ret_prob"]]
bars = ax.barh(fsort["symbol"], fsort["avg_ret_prob"], color=colors, alpha=0.85)
ax.axvline(0.65, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Predicted Retention Probability")
ax.set_title(f"Predicted Retention Probability — Next KSE-30 Rebalancing\n"
             f"(~{pred_target.strftime('%b %Y')})")
for bar, val in zip(bars, fsort["avg_ret_prob"]):
    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8)
plt.tight_layout()
savefig("rebalancing","R01_retention_probability.png")

# Feature importances
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle("Feature Importances — Rebalancing Models", fontweight="bold")
fi_r = pd.DataFrame({"Feature":FEAT_COLS,
                      "Imp":rf_r.feature_importances_}).sort_values("Imp")
fi_c = pd.DataFrame({"Feature":FEAT_COLS,
                      "Imp":rf_c.feature_importances_}).sort_values("Imp")
axes[0].barh(fi_r["Feature"], fi_r["Imp"], color="#3498db", alpha=0.85)
axes[0].set_title("Weight Prediction (RF)")
axes[1].barh(fi_c["Feature"], fi_c["Imp"], color="#e74c3c", alpha=0.85)
axes[1].set_title("Inclusion Prediction (RF)")
plt.tight_layout()
savefig("rebalancing","R02_feature_importances.png")

# Weight scatter: actual vs predicted
test_ret_p = test_p[test_p.retained==1].copy()
test_ret_p["pred_wt"] = ridge.predict(
    scaler.transform(test_ret_p[FEAT_COLS].values))
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(test_ret_p["target_weight"], test_ret_p["pred_wt"],
           alpha=0.7, color="#3498db", s=55)
lim = max(test_ret_p["target_weight"].max(), test_ret_p["pred_wt"].max())*1.1
ax.plot([0,lim],[0,lim],"k--",linewidth=1)
ax.set_xlabel("Actual weight (%)"); ax.set_ylabel("Predicted weight (%)")
ax.set_title(f"Weight Prediction — Test Set\nRidge R²={m_ridge_wt['R2']:.4f}")
plt.tight_layout()
savefig("rebalancing","R03_weight_scatter.png")

# Predicted weight changes for next rebalancing
fig, ax = plt.subplots(figsize=(12,5))
fwc = future_df.copy()
fwc["wt_chg"] = fwc["pred_wt_avg"] - fwc["cur_weight"]
fwc = fwc.sort_values("wt_chg")
cols = ["#e74c3c" if v<0 else "#2ecc71" for v in fwc["wt_chg"]]
ax.barh(fwc["symbol"], fwc["wt_chg"], color=cols, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Predicted Weight Change (%)")
ax.set_title("Predicted Weight Changes at Next Rebalancing")
plt.tight_layout()
savefig("rebalancing","R04_weight_changes.png")

# Save rebalancing results
reb_rows = [
    {"Task":"Weight","Model":"Naive","RMSE":round(m_naive_wt["RMSE"],4),
     "MAE":round(m_naive_wt["MAE"],4),"R2":round(m_naive_wt["R2"],4)},
    {"Task":"Weight","Model":"Ridge","RMSE":round(m_ridge_wt["RMSE"],4),
     "MAE":round(m_ridge_wt["MAE"],4),"R2":round(m_ridge_wt["R2"],4)},
    {"Task":"Weight","Model":"RandomForest","RMSE":round(m_rf_wt["RMSE"],4),
     "MAE":round(m_rf_wt["MAE"],4),"R2":round(m_rf_wt["R2"],4)},
    {"Task":"Inclusion","Model":"Naive","Accuracy":round(m_naive_clf["Accuracy"],4),
     "AUC":round(m_naive_clf["AUC"],4)},
    {"Task":"Inclusion","Model":"Logistic","Accuracy":round(m_logit["Accuracy"],4),
     "AUC":round(m_logit["AUC"],4)},
    {"Task":"Inclusion","Model":"RandomForest","Accuracy":round(m_rf_clf["Accuracy"],4),
     "AUC":round(m_rf_clf["AUC"],4)},
]
pd.DataFrame(reb_rows).to_csv(
    os.path.join(BASE,"results_rebalancing.csv"), index=False)
future_df.to_csv(
    os.path.join(BASE,"results_rebalancing_forecast.csv"), index=False)
print("\n  Saved: results_rebalancing.csv, results_rebalancing_forecast.csv")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SECTION 8 — Results summary")
print("="*65)

print("\n[Table 1] Descriptive statistics — Daily NAV returns")
print(desc_df.to_string(index=False))

print("\n[Table 2] GARCH model selection")
print(pd.DataFrame(garch_rows).to_string(index=False))

print("\n[Table 3] Fund flow prediction")
ff_df = pd.read_csv(os.path.join(BASE,"results_fund_flow.csv"))
print(ff_df[ff_df.Target=="Total"].to_string(index=False))

print("\n[Table 4] Market efficiency")
eff_out = pd.DataFrame(eff_rows)
print(eff_out[["Fund","Runs Z","Runs p","Runs verdict",
               "VR(2)","VR(2) p","LB Q p","Hurst H"]].to_string(index=False))

print("\n[Table 5] Weight prediction")
reb_df = pd.read_csv(os.path.join(BASE,"results_rebalancing.csv"))
print(reb_df[reb_df.Task=="Weight"].to_string(index=False))

print("\n[Table 6] Inclusion prediction")
print(reb_df[reb_df.Task=="Inclusion"].to_string(index=False))

print("\n[Table 7] At-risk stocks — next rebalancing")
at_risk = future_df[future_df["exclusion_risk"]>0.20].sort_values(
    "exclusion_risk", ascending=False
)[["symbol","cur_weight","avg_ret_prob","exclusion_risk","pred_wt_avg"]].round(4)
print(at_risk.to_string(index=False))

# Summary dashboard figure
fig = plt.figure(figsize=(16,10))
fig.suptitle("KSE-30 Fund Flow & Rebalancing Analysis — Results Dashboard",
             fontsize=14, fontweight="bold", y=0.98)
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# Panel 1: GARCH persistence
ax1 = fig.add_subplot(gs[0,0])
funds_g = ["AKD","NBP","NIT"]
persist = [next(r["persist"] for r in garch_rows if r["Fund"]==f) for f in funds_g]
colors_g = ["#e74c3c" if p>0.95 else "#f39c12" if p>0.90 else "#2ecc71"
            for p in persist]
ax1.bar(funds_g, persist, color=colors_g, alpha=0.85)
ax1.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
ax1.axhline(0.95, color="orange", linewidth=0.8, linestyle=":")
ax1.set_ylim(0, 1.05); ax1.set_title("GARCH(1,1)\nPersistence (α+β)")
for i,(f,p) in enumerate(zip(funds_g,persist)):
    ax1.text(i,p+0.005,f"{p:.3f}",ha="center",fontsize=9)

# Panel 2: fund flow model comparison (directional accuracy)
ax2 = fig.add_subplot(gs[0,1])
models_ff = ["Naive","ARIMAX","VAR(1)"]
da_vals   = [m_naive["DirAcc"], m_arimax["DirAcc"], m_var["DirAcc"]]
ax2.bar(models_ff, da_vals, color=["#95a5a6","#e74c3c","#2ecc71"], alpha=0.85)
ax2.axhline(50, color="black", linewidth=1, linestyle="--", label="50% (chance)")
ax2.set_title("Fund Flow Prediction\nDirectional Accuracy (%)")
ax2.set_ylabel("%"); ax2.legend(fontsize=8)
for i,(m,v) in enumerate(zip(models_ff,da_vals)):
    ax2.text(i,v+0.5,f"{v:.1f}%",ha="center",fontsize=9)

# Panel 3: efficiency test summary
ax3 = fig.add_subplot(gs[0,2])
test_labels = ["Runs\n(p<0.05?)", "VR(2)\n(p<0.05?)", "LB Q\n(p<0.05?)",
               "Hurst\n(H>0.55?)"]
x = np.arange(len(test_labels)); width = 0.25
for fi, fund in enumerate(funds_g):
    row = next(r for r in eff_rows if r["Fund"]==fund)
    vals = [1 if row["Runs p"]<0.05 else 0,
            1 if row["VR(2) p"]<0.05 else 0,
            1 if row["LB Q p"]<0.05 else 0,
            1 if (row["Hurst H"] or 0)>0.55 else 0]
    ax3.bar(x+fi*width, vals, width, color=list(FUND_COLORS.values())[fi],
            alpha=0.85, label=fund)
ax3.set_xticks(x+width); ax3.set_xticklabels(test_labels, fontsize=8)
ax3.set_yticks([0,1]); ax3.set_yticklabels(["Efficient","Inefficient"])
ax3.set_title("Market Efficiency\nby Test & Fund")
ax3.legend(fontsize=8)

# Panel 4: total fund flow
ax4 = fig.add_subplot(gs[1,:2])
bar_c = ["#2980b9" if v>=0 else "#e74c3c" for v in monthly["total_fund_flow"]]
ax4.bar(monthly["date"], monthly["total_fund_flow"], color=bar_c, width=20, alpha=0.85)
ax4.axhline(0, color="black", linewidth=0.6)
ax4.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", linewidth=1.5)
ax4.plot(dates_te, np.array(var_preds), "o-", color="#2ecc71",
         linewidth=2, markersize=5, label="VAR(1) forecast")
ax4.set_title("Total Fund Flow (PKR Millions) with VAR(1) Forecast")
ax4.set_ylabel("PKR mn"); ax4.legend(fontsize=8)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)

# Panel 5: at-risk stocks
ax5 = fig.add_subplot(gs[1,2])
at_risk_plot = future_df.sort_values("avg_ret_prob").head(10)
colors_ar = ["#e74c3c" if p<0.65 else "#f39c12" if p<0.80 else "#2ecc71"
             for p in at_risk_plot["avg_ret_prob"]]
ax5.barh(at_risk_plot["symbol"], at_risk_plot["avg_ret_prob"],
         color=colors_ar, alpha=0.85)
ax5.axvline(0.65, color="black", linewidth=1, linestyle="--")
ax5.set_xlabel("Retention prob.")
ax5.set_title("Bottom 10 Stocks\n(Retention Probability)")

savefig("summary","SUMMARY_dashboard.png")

print("\n" + "="*65)
print("PIPELINE COMPLETE")
print("="*65)
print(f"\nOutputs:")
print(f"  daily_master.csv")
print(f"  monthly_master.csv")
print(f"  kse30_stocks_clean.csv")
print(f"  results_garch.csv")
print(f"  results_fund_flow.csv")
print(f"  results_efficiency.csv")
print(f"  results_rebalancing.csv")
print(f"  results_rebalancing_forecast.csv")
print(f"  figures/  ({sum(1 for _ in __import__('pathlib').Path(FIG_BASE).rglob('*.png'))} figures)")
