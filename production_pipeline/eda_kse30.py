"""
Exploratory data analysis for the raw KSE-30 constituent panels.

Inputs:
  - kse30_daily_data.csv
  - kse30_stocks_clean.csv

Outputs:
  - Console summaries
  - PNG charts under output/eda_raw/
  - summary CSV tables under output/eda_raw/

Run:
  python production_pipeline/eda_kse30.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ANALYSIS_DIR = BASE_DIR / "output" / "analysis"
RAW_PATH = RAW_DIR / "kse30_daily_data.csv"
CLEAN_PATH = ANALYSIS_DIR / "kse30_stocks_clean.csv"
FUNDS_PATH = RAW_DIR / "funds_data.xlsx"
MACRO_PATH = RAW_DIR / "macro_data.xlsx"
CPI_PATH = RAW_DIR / "cpi.csv"
OUTPUT_DIR = BASE_DIR / "output" / "eda_raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


RAW_RENAME = {
    "Date": "date",
    "ISIN": "isin",
    "SYMBOL": "symbol",
    "COMPANY": "company",
    "PRICE": "price",
    "IDX WT %": "weight_pct",
    "FF BASED SHARES": "ff_shares",
    "FF BASED MCAP": "ff_mcap",
    "ORD SHARES": "ord_shares",
    "ORD SHARES MCAP": "ord_mcap",
    "VOLUME": "volume",
}

NUMERIC_COLS = [
    "price",
    "weight_pct",
    "ff_shares",
    "ff_mcap",
    "ord_shares",
    "ord_mcap",
    "volume",
]


def read_panel(path, rename=None):
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)

    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["company"] = df["company"].astype(str).str.strip()

    for col in [c for c in df.columns if c not in {"date", "isin", "symbol", "company"}]:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def read_funds(path):
    funds = {}
    for sheet in ["AKD", "NBP", "NTI"]:
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.rename(columns={"DATE": "date", "NAV": "nav", "AUM": "aum"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["aum"] = pd.to_numeric(df["aum"], errors="coerce")
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
        df["log_return"] = np.log(df["nav"] / df["nav"].shift(1))
        df["rolling_vol_30d"] = df["log_return"].rolling(30, min_periods=15).std() * np.sqrt(252)
        funds[sheet] = df
    return funds


def compute_monthly_flows(df):
    monthly = (
        df.set_index("date")
        .resample("ME")
        .agg(nav_start=("nav", "first"), nav_end=("nav", "last"), aum=("aum", "last"))
        .reset_index()
    )
    monthly["aum_prev"] = monthly["aum"].shift(1)
    monthly["nav_return_m"] = monthly["nav_end"] / monthly["nav_start"] - 1
    monthly["fund_flow"] = monthly["aum"] - monthly["aum_prev"] * (1 + monthly["nav_return_m"])
    monthly["fund_flow_pct"] = monthly["fund_flow"] / monthly["aum_prev"]
    return monthly.dropna(subset=["fund_flow"]).reset_index(drop=True)


def savefig(name):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def dataset_overview(name, df):
    print_section(f"{name} OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Symbols: {df['symbol'].nunique():,}")
    print(f"Trading dates: {df['date'].nunique():,}")
    print("\nColumns:")
    print(", ".join(df.columns))

    missing = (
        df.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .assign(percent=lambda x: (x["missing"] / len(df) * 100).round(2))
    )
    missing = missing[missing["missing"] > 0]
    print("\nMissing values:")
    print(missing if not missing.empty else "No missing values")


def save_summary_tables(raw, clean):
    raw_summary = raw[[c for c in NUMERIC_COLS if c in raw.columns]].describe().T.round(4)
    clean_cols = [c for c in NUMERIC_COLS + ["log_return", "rolling_vol_30d", "ma_20", "ma_50"] if c in clean.columns]
    clean_summary = clean[clean_cols].describe().T.round(4)

    raw_summary.to_csv(OUTPUT_DIR / "raw_numeric_summary.csv")
    clean_summary.to_csv(OUTPUT_DIR / "clean_numeric_summary.csv")

    symbol_summary = (
        clean.groupby("symbol")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            observations=("date", "count"),
            avg_price=("price", "mean"),
            avg_weight_pct=("weight_pct", "mean"),
            avg_volume=("volume", "mean"),
            avg_daily_return=("log_return", "mean"),
            daily_volatility=("log_return", "std"),
            avg_rolling_vol_30d=("rolling_vol_30d", "mean"),
        )
        .sort_values("avg_weight_pct", ascending=False)
    )
    symbol_summary.to_csv(OUTPUT_DIR / "clean_symbol_summary.csv")

    print_section("SUMMARY TABLES")
    print(f"Saved raw_numeric_summary.csv, clean_numeric_summary.csv, clean_symbol_summary.csv to {OUTPUT_DIR}")
    print("\nTop 10 symbols by average cleaned index weight:")
    print(symbol_summary.head(10).round(4))


def plot_raw_coverage(raw, clean):
    raw_counts = raw.groupby("date")["symbol"].nunique().rename("raw")
    clean_counts = clean.groupby("date")["symbol"].nunique().rename("clean")
    coverage = pd.concat([raw_counts, clean_counts], axis=1).sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(coverage.index, coverage["raw"], label="Raw daily data", linewidth=1.2)
    ax.plot(coverage.index, coverage["clean"], label="Clean stock panel", linewidth=1.2)
    ax.set_title("KSE-30 Constituent Coverage by Date")
    ax.set_ylabel("Number of symbols")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30)
    savefig("01_constituent_coverage.png")


def plot_index_weight_snapshot(raw, clean):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=False)
    panels = [("Raw", raw), ("Clean", clean)]

    for ax, (label, df) in zip(axes, panels):
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].sort_values("weight_pct", ascending=True)
        ax.barh(latest["symbol"], latest["weight_pct"], color="#4c78a8")
        ax.set_title(f"{label}: Constituent Weights on {latest_date.date()}")
        ax.set_xlabel("Index weight (%)")
        ax.set_ylabel("")

    savefig("02_latest_index_weights.png")


def plot_price_distributions(raw, clean):
    latest_raw = raw[raw["date"] == raw["date"].max()]
    latest_clean = clean[clean["date"] == clean["date"].max()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(latest_raw["price"].dropna(), bins=25, edgecolor="white", color="#72b7b2")
    axes[0].set_title(f"Raw Price Distribution ({raw['date'].max().date()})")
    axes[0].set_xlabel("Price (PKR)")
    axes[0].set_ylabel("Symbols")

    axes[1].hist(latest_clean["price"].dropna(), bins=25, edgecolor="white", color="#f58518")
    axes[1].set_title(f"Clean Price Distribution ({clean['date'].max().date()})")
    axes[1].set_xlabel("Price (PKR)")
    axes[1].set_ylabel("Symbols")
    savefig("03_latest_price_distribution.png")


def plot_aggregate_market(raw):
    daily = (
        raw.groupby("date")
        .agg(total_ff_mcap=("ff_mcap", "sum"), total_volume=("volume", "sum"), total_weight=("weight_pct", "sum"))
        .reset_index()
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(daily["date"], daily["total_ff_mcap"] / 1e12, color="#4c78a8", linewidth=1)
    axes[0].set_title("Total Free-Float Market Cap")
    axes[0].set_ylabel("PKR tn")

    axes[1].plot(daily["date"], daily["total_volume"] / 1e6, color="#54a24b", linewidth=1)
    axes[1].set_title("Total Daily Volume")
    axes[1].set_ylabel("Shares mn")

    axes[2].plot(daily["date"], daily["total_weight"], color="#e45756", linewidth=1)
    axes[2].axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[2].set_title("Reported Sum of Index Weights")
    axes[2].set_ylabel("Percent")
    axes[2].set_xlabel("Date")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30)
    savefig("04_raw_aggregate_market_timeseries.png")


def plot_clean_return_distribution(clean):
    returns = clean["log_return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(returns, bins=80, color="#4c78a8", edgecolor="white", density=True)
    axes[0].axvline(returns.mean(), color="black", linestyle="--", linewidth=1, label="Mean")
    axes[0].set_title("Clean Panel Daily Log Return Distribution")
    axes[0].set_xlabel("Daily log return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sns.boxplot(data=clean, x="log_return", ax=axes[1], color="#f58518")
    axes[1].set_title("Daily Log Return Box Plot")
    axes[1].set_xlabel("Daily log return")
    savefig("05_clean_log_return_distribution.png")

    print_section("CLEAN RETURN SUMMARY")
    print(returns.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(6))
    print(f"Skew: {returns.skew():.4f}")
    print(f"Excess kurtosis: {returns.kurtosis():.4f}")


def plot_top_symbol_prices(clean):
    latest = clean[clean["date"] == clean["date"].max()]
    top_symbols = latest.nlargest(8, "weight_pct")["symbol"].tolist()
    panel = clean[clean["symbol"].isin(top_symbols)].copy()

    fig, ax = plt.subplots(figsize=(13, 6))
    for symbol, group in panel.groupby("symbol"):
        normalized = group["price"] / group["price"].dropna().iloc[0] * 100
        ax.plot(group["date"], normalized, linewidth=1.2, label=symbol)

    ax.set_title("Normalized Prices for Top Clean Constituents by Latest Weight")
    ax.set_ylabel("Indexed price, first observation = 100")
    ax.set_xlabel("Date")
    ax.legend(ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    savefig("06_top_symbol_normalized_prices.png")


def plot_volatility_leaders(clean):
    vol = (
        clean.groupby("symbol")["log_return"]
        .std()
        .mul(np.sqrt(252))
        .dropna()
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vol.index, vol.values, color="#e45756")
    ax.set_title("Highest Annualized Volatility by Symbol")
    ax.set_xlabel("Annualized volatility")
    savefig("07_volatility_leaders.png")


def plot_rolling_volatility(clean):
    latest = clean[clean["date"] == clean["date"].max()]
    top_symbols = latest.nlargest(6, "weight_pct")["symbol"].tolist()
    panel = clean[clean["symbol"].isin(top_symbols)]

    fig, ax = plt.subplots(figsize=(13, 5))
    for symbol, group in panel.groupby("symbol"):
        ax.plot(group["date"], group["rolling_vol_30d"], linewidth=1, label=symbol)

    ax.set_title("30-Day Rolling Volatility for Top Clean Constituents")
    ax.set_ylabel("Rolling volatility")
    ax.set_xlabel("Date")
    ax.legend(ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    savefig("08_rolling_volatility_top_symbols.png")


def plot_returns_heatmap(clean):
    pivot = clean.pivot_table(index="date", columns="symbol", values="log_return")
    valid_cols = pivot.notna().sum().sort_values(ascending=False).head(30).index
    corr = pivot[valid_cols].corr(min_periods=40)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.2,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Clean Panel Return Correlations (Top 30 by Observations)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig("09_clean_return_correlation_heatmap.png")


def plot_liquidity_vs_weight(clean):
    latest = clean[clean["date"] == clean["date"].max()][["symbol", "weight_pct"]]
    liquidity = (
        clean.groupby("symbol")
        .agg(avg_volume=("volume", "mean"), avg_ff_mcap=("ff_mcap", "mean"))
        .reset_index()
    )
    merged = latest.merge(liquidity, on="symbol", how="left").dropna(subset=["avg_volume", "avg_ff_mcap"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = np.clip(merged["avg_ff_mcap"] / merged["avg_ff_mcap"].median() * 55, 20, 300)
    ax.scatter(merged["weight_pct"], merged["avg_volume"] / 1e6, s=sizes, alpha=0.65, color="#4c78a8")
    for _, row in merged.nlargest(10, "weight_pct").iterrows():
        ax.annotate(row["symbol"], (row["weight_pct"], row["avg_volume"] / 1e6), fontsize=8)
    ax.set_title("Latest Weight vs Average Daily Volume")
    ax.set_xlabel("Latest index weight (%)")
    ax.set_ylabel("Average volume (shares mn)")
    savefig("10_liquidity_vs_weight.png")


def plot_missingness(clean):
    miss = clean.isna().mean().mul(100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(miss.index, miss.values, color="#bab0ac")
    ax.set_title("Clean Panel Missing Values by Column")
    ax.set_xlabel("Missing values (%)")
    savefig("11_clean_missingness.png")


def funds_overview(funds):
    print_section("FUNDS DATA OVERVIEW")
    for name, df in funds.items():
        print(
            f"{name}: rows={len(df):,} | {df['date'].min().date()} to {df['date'].max().date()} | "
            f"missing NAV={df['nav'].isna().sum()} | missing AUM={df['aum'].isna().sum()}"
        )


def save_funds_tables(funds):
    rows = []
    monthly_rows = []
    for fund, df in funds.items():
        d = df.copy()
        rows.append(
            {
                "fund": fund,
                "start_date": d["date"].min(),
                "end_date": d["date"].max(),
                "rows": len(d),
                "mean_nav": d["nav"].mean(),
                "std_nav": d["nav"].std(),
                "mean_aum": d["aum"].mean(),
                "std_aum": d["aum"].std(),
                "mean_daily_return": d["log_return"].mean(),
                "daily_volatility": d["log_return"].std(),
            }
        )
        m = compute_monthly_flows(d)
        monthly_rows.append(
            {
                "fund": fund,
                "months": len(m),
                "mean_monthly_flow": m["fund_flow"].mean(),
                "std_monthly_flow": m["fund_flow"].std(),
                "positive_flow_months": int((m["fund_flow"] > 0).sum()),
                "negative_flow_months": int((m["fund_flow"] < 0).sum()),
            }
        )
        m.to_csv(OUTPUT_DIR / f"funds_{fund.lower()}_monthly_flows.csv", index=False)

    funds_summary = pd.DataFrame(rows).round(6)
    monthly_summary = pd.DataFrame(monthly_rows).round(6)
    funds_summary.to_csv(OUTPUT_DIR / "funds_summary.csv", index=False)
    monthly_summary.to_csv(OUTPUT_DIR / "funds_monthly_flow_summary.csv", index=False)
    print("Saved funds_summary.csv, funds_monthly_flow_summary.csv, and per-fund monthly flow CSVs")


def plot_funds_nav_aum(funds):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}

    for fund, df in funds.items():
        axes[0].plot(df["date"], df["nav"], linewidth=1.2, label=fund, color=colors[fund])
        axes[1].plot(df["date"], df["aum"], linewidth=1.2, label=fund, color=colors[fund])

    axes[0].set_title("Funds NAV Time Series")
    axes[0].set_ylabel("NAV")
    axes[0].legend()
    axes[1].set_title("Funds AUM Time Series")
    axes[1].set_ylabel("AUM")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=30)
    savefig("12_funds_nav_aum_timeseries.png")


def plot_funds_return_distributions(funds):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
    for i, fund in enumerate(["AKD", "NBP", "NTI"]):
        r = funds[fund]["log_return"].dropna()
        axes[i].hist(r, bins=60, density=True, color=colors[fund], edgecolor="white")
        axes[i].axvline(r.mean(), color="black", linestyle="--", linewidth=1)
        axes[i].set_title(f"{fund} log returns")
        axes[i].set_xlabel("Daily log return")
    axes[0].set_ylabel("Density")
    savefig("13_funds_log_return_distributions.png")


def plot_funds_rolling_volatility(funds):
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
    for fund, df in funds.items():
        ax.plot(df["date"], df["rolling_vol_30d"], linewidth=1.1, label=fund, color=colors[fund])
    ax.set_title("Funds 30-Day Rolling Volatility (Annualized)")
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=30)
    savefig("14_funds_rolling_volatility.png")


def plot_funds_monthly_flows(funds):
    monthly = []
    for fund, df in funds.items():
        m = compute_monthly_flows(df)
        m["fund"] = fund
        monthly.append(m)
    all_m = pd.concat(monthly, ignore_index=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
    for idx, fund in enumerate(["AKD", "NBP", "NTI"]):
        s = all_m[all_m["fund"] == fund]
        bar_c = np.where(s["fund_flow"] >= 0, colors[fund], "#d62728")
        axes[idx].bar(s["date"], s["fund_flow"], color=bar_c, width=20, alpha=0.85)
        axes[idx].axhline(0, color="black", linewidth=0.8)
        axes[idx].set_title(f"{fund} monthly fund flow")
        axes[idx].set_ylabel("Flow")
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=30)
    savefig("15_funds_monthly_flows.png")

    pivot = all_m.pivot_table(index="date", columns="fund", values="fund_flow")
    corr = pivot.corr(min_periods=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Monthly Fund Flow Correlation")
    savefig("16_funds_flow_correlation.png")


def plot_funds_missingness(funds):
    rows = []
    for fund, df in funds.items():
        for col in ["nav", "aum", "log_return", "rolling_vol_30d"]:
            rows.append({"fund": fund, "column": col, "missing_pct": df[col].isna().mean() * 100})
    md = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=md, x="column", y="missing_pct", hue="fund", ax=ax)
    ax.set_title("Funds Missingness by Column")
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("")
    savefig("17_funds_missingness.png")


def read_macro_cpi(macro_path, cpi_path):
    oil = pd.read_excel(macro_path, sheet_name="OIL").rename(columns={"DATE": "date", "PRICE": "oil_price"})
    ir = pd.read_excel(macro_path, sheet_name="IR").rename(columns={"DATE": "date", "RATE": "interest_rate"})
    usd = pd.read_excel(macro_path, sheet_name="USD").rename(columns={"DATE": "date", "USD": "usdpkr"})

    for df in [oil, ir, usd]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    oil["oil_price"] = pd.to_numeric(oil["oil_price"], errors="coerce")
    ir["interest_rate"] = pd.to_numeric(ir["interest_rate"], errors="coerce")
    usd["usdpkr"] = pd.to_numeric(usd["usdpkr"], errors="coerce")

    oil = oil.sort_values("date").drop_duplicates(subset="date", keep="last")
    ir = ir.sort_values("date").drop_duplicates(subset="date", keep="last")
    usd = usd.sort_values("date").drop_duplicates(subset="date", keep="last")

    oil["oil_log_return"] = np.log(oil["oil_price"] / oil["oil_price"].shift(1))
    usd["usdpkr_log_return"] = np.log(usd["usdpkr"] / usd["usdpkr"].shift(1))

    cpi = pd.read_csv(cpi_path, skiprows=1, header=0)
    cpi.columns = ["period_str", "cpi_yoy"]
    cpi = cpi.dropna()
    cpi["cpi_yoy"] = pd.to_numeric(cpi["cpi_yoy"], errors="coerce")

    month_map = {
        "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
        "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
        "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec",
    }

    def normalize_month(x):
        s = str(x)
        for full, abbr in month_map.items():
            if s.startswith(full):
                return s.replace(full, abbr, 1)
        return s

    cpi["period_str"] = cpi["period_str"].apply(normalize_month)
    cpi["date"] = pd.to_datetime(cpi["period_str"], format="%b-%y", errors="coerce") + pd.offsets.MonthEnd(0)
    cpi = cpi[["date", "cpi_yoy"]].dropna().sort_values("date").reset_index(drop=True)

    return oil.reset_index(drop=True), ir.reset_index(drop=True), usd.reset_index(drop=True), cpi


def save_macro_summary(oil, ir, usd, cpi):
    summary = pd.DataFrame(
        [
            {"series": "oil_price", "start": oil["date"].min(), "end": oil["date"].max(), "n_obs": len(oil), "mean": oil["oil_price"].mean(), "std": oil["oil_price"].std()},
            {"series": "interest_rate", "start": ir["date"].min(), "end": ir["date"].max(), "n_obs": len(ir), "mean": ir["interest_rate"].mean(), "std": ir["interest_rate"].std()},
            {"series": "usdpkr", "start": usd["date"].min(), "end": usd["date"].max(), "n_obs": len(usd), "mean": usd["usdpkr"].mean(), "std": usd["usdpkr"].std()},
            {"series": "cpi_yoy", "start": cpi["date"].min(), "end": cpi["date"].max(), "n_obs": len(cpi), "mean": cpi["cpi_yoy"].mean(), "std": cpi["cpi_yoy"].std()},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "macro_cpi_summary.csv", index=False)
    print("Saved macro_cpi_summary.csv")


def plot_macro_levels(oil, ir, usd, cpi):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=False)

    axes[0].plot(oil["date"], oil["oil_price"], color="#c0392b", linewidth=1.2)
    axes[0].set_title("Brent Oil Price")
    axes[0].set_ylabel("USD/bbl")

    axes[1].plot(usd["date"], usd["usdpkr"], color="#8e44ad", linewidth=1.2)
    axes[1].set_title("USD/PKR")
    axes[1].set_ylabel("PKR per USD")

    axes[2].step(ir["date"], ir["interest_rate"], where="post", color="#16a085", linewidth=1.3)
    axes[2].set_title("Policy Interest Rate")
    axes[2].set_ylabel("%")

    axes[3].plot(cpi["date"], cpi["cpi_yoy"], color="#2d3436", linewidth=1.3)
    axes[3].set_title("CPI YoY")
    axes[3].set_ylabel("%")
    axes[3].set_xlabel("Date")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.autofmt_xdate(rotation=30)
    savefig("18_macro_cpi_levels.png")


def plot_macro_returns(oil, usd):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    axes[0].plot(oil["date"], oil["oil_log_return"], color="#c0392b", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Oil Daily Log Return")
    axes[0].set_ylabel("Log return")

    axes[1].plot(usd["date"], usd["usdpkr_log_return"], color="#8e44ad", linewidth=0.9)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("USD/PKR Daily Log Return")
    axes[1].set_ylabel("Log return")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.autofmt_xdate(rotation=30)
    savefig("19_macro_log_returns.png")


def plot_macro_cpi_correlation(oil, ir, usd, cpi):
    monthly = (
        oil.set_index("date")[["oil_log_return"]]
        .resample("ME").sum()
        .rename(columns={"oil_log_return": "oil_return_monthly"})
    )
    usd_m = (
        usd.set_index("date")[["usdpkr_log_return"]]
        .resample("ME").sum()
        .rename(columns={"usdpkr_log_return": "usdpkr_return_monthly"})
    )
    ir_m = ir.set_index("date")[["interest_rate"]].resample("ME").last().rename(columns={"interest_rate": "interest_rate_end"})
    cpi_m = cpi.set_index("date")[["cpi_yoy"]]

    m = monthly.join([usd_m, ir_m, cpi_m], how="inner").dropna()
    corr = m.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Monthly Macro/CPI Correlation")
    savefig("20_macro_cpi_correlation.png")

    m.to_csv(OUTPUT_DIR / "macro_cpi_monthly_panel.csv", index=True)
    print("Saved macro_cpi_monthly_panel.csv")


def main():
    print("Loading KSE-30 raw and cleaned data from the production workspace...")
    raw = read_panel(RAW_PATH, RAW_RENAME)
    clean = read_panel(CLEAN_PATH)

    dataset_overview("RAW DAILY DATA", raw)
    dataset_overview("CLEAN STOCK PANEL", clean)
    save_summary_tables(raw, clean)

    print_section("GENERATING FIGURES")
    plot_raw_coverage(raw, clean)
    plot_index_weight_snapshot(raw, clean)
    plot_price_distributions(raw, clean)
    plot_aggregate_market(raw)
    plot_clean_return_distribution(clean)
    plot_top_symbol_prices(clean)
    plot_volatility_leaders(clean)
    plot_rolling_volatility(clean)
    plot_returns_heatmap(clean)
    plot_liquidity_vs_weight(clean)
    plot_missingness(clean)

    print_section("FUNDS EDA")
    funds = read_funds(FUNDS_PATH)
    funds_overview(funds)
    save_funds_tables(funds)
    plot_funds_nav_aum(funds)
    plot_funds_return_distributions(funds)
    plot_funds_rolling_volatility(funds)
    plot_funds_monthly_flows(funds)
    plot_funds_missingness(funds)

    if MACRO_PATH.exists() and CPI_PATH.exists():
        print_section("MACRO + CPI EDA")
        oil, ir, usd, cpi = read_macro_cpi(MACRO_PATH, CPI_PATH)
        save_macro_summary(oil, ir, usd, cpi)
        plot_macro_levels(oil, ir, usd, cpi)
        plot_macro_returns(oil, usd)
        plot_macro_cpi_correlation(oil, ir, usd, cpi)
    else:
        print_section("MACRO + CPI EDA")
        print(f"Skipped macro/CPI plots. Missing files: {MACRO_PATH} or {CPI_PATH}")

    print(f"\nDone. Figures and summary tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
