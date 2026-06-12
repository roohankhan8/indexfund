"""
Generate report-ready EDA figures for Chapter 3 (Methodology).

Sources:
  - production_pipeline/data/raw/kse30_daily_data.csv
  - production_pipeline/output/analysis/kse30_stocks_clean.csv
  - production_pipeline/data/raw/funds_data.xlsx

Outputs:
  - docs/report_workspace/chapter-03-methodology/images/C3_EDA_*.png
  - docs/report_workspace/chapter-03-methodology/images/C3_EDA_summary.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ANALYSIS_DIR = BASE_DIR / "output" / "analysis"
OUT_DIR = BASE_DIR.parent / "docs" / "report_workspace" / "chapter-03-methodology" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / "kse30_daily_data.csv"
CLEAN_PATH = ANALYSIS_DIR / "kse30_stocks_clean.csv"
FUNDS_PATH = RAW_DIR / "funds_data.xlsx"


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def savefig(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"saved -> {path}")


def load_data():
    raw = pd.read_csv(RAW_PATH).rename(
        columns={
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
    )
    raw["date"] = pd.to_datetime(raw["date"])
    raw["symbol"] = raw["symbol"].astype(str).str.upper().str.strip()
    for col in ["price", "weight_pct", "ff_shares", "ff_mcap", "ord_shares", "ord_mcap", "volume"]:
        if col in raw.columns:
            if raw[col].dtype == "object":
                raw[col] = raw[col].str.replace(",", "", regex=False)
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    clean = pd.read_csv(CLEAN_PATH)
    clean["date"] = pd.to_datetime(clean["date"])
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    for col in ["price", "weight_pct", "ff_shares", "ff_mcap", "ord_shares", "ord_mcap", "volume", "log_return"]:
        if col in clean.columns:
            if clean[col].dtype == "object":
                clean[col] = clean[col].str.replace(",", "", regex=False)
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    funds = {}
    for sheet in ["AKD", "NBP", "NTI"]:
        d = pd.read_excel(FUNDS_PATH, sheet_name=sheet).rename(
            columns={"DATE": "date", "NAV": "nav", "AUM": "aum"}
        )
        d["date"] = pd.to_datetime(d["date"])
        d["nav"] = pd.to_numeric(d["nav"], errors="coerce")
        d["aum"] = pd.to_numeric(d["aum"], errors="coerce")
        d = d.sort_values("date").drop_duplicates("date", keep="last")
        d["log_return"] = np.log(d["nav"] / d["nav"].shift(1))
        funds[sheet] = d

    return raw, clean, funds


def graph_01_coverage_and_quality(raw, clean):
    by_date = pd.DataFrame(
        {
            "raw_constituents": raw.groupby("date")["symbol"].nunique(),
            "clean_constituents": clean.groupby("date")["symbol"].nunique(),
            "raw_non_missing_volume_pct": raw.groupby("date")["volume"].apply(
                lambda s: s.notna().mean() * 100
            ),
            "raw_zero_or_missing_volume_pct": raw.groupby("date")["volume"].apply(
                lambda s: (s.isna() | s.eq(0)).mean() * 100
            ),
        }
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(by_date["date"], by_date["raw_constituents"], label="Raw", lw=1.4)
    axes[0].plot(by_date["date"], by_date["clean_constituents"], label="Clean", lw=1.2)
    axes[0].set_title("KSE-30 Constituent Count by Date")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="lower right")

    axes[1].plot(by_date["date"], by_date["raw_non_missing_volume_pct"], lw=1.4, color="#2ca02c", label="Volume available")
    axes[1].plot(by_date["date"], by_date["raw_zero_or_missing_volume_pct"], lw=1.2, color="#d62728", label="Volume zero/missing")
    axes[1].set_title("Volume Data Availability by Date (Raw Panel)")
    axes[1].set_ylabel("Percent of constituents")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="center right")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30)
    savefig("C3_EDA_01_coverage_and_volume_quality.png")


def graph_02_volume_gap_heatmap(raw):
    d = raw.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["bad_volume"] = (d["volume"].isna() | d["volume"].eq(0)).astype(int)
    hm = d.groupby(["year", "month"])["bad_volume"].mean().unstack()
    hm = hm.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(hm * 100, cmap="YlOrRd", annot=True, fmt=".0f", cbar_kws={"label": "% zero/missing volume"}, ax=ax)
    ax.set_title("Monthly Concentration of Zero/Missing Volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    savefig("C3_EDA_02_volume_gap_heatmap.png")


def graph_03_market_aggregates(raw):
    daily = raw.groupby("date", as_index=False).agg(
        total_ff_mcap=("ff_mcap", "sum"),
        total_volume=("volume", "sum"),
        total_weight=("weight_pct", "sum"),
    )
    gap = daily["total_volume"].eq(0) | daily["total_volume"].isna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    axes[0].plot(daily["date"], daily["total_ff_mcap"] / 1e12, color="#1f77b4", lw=1.2)
    axes[0].set_title("Aggregate Free-Float Market Cap")
    axes[0].set_ylabel("PKR tn")

    axes[1].plot(daily["date"], daily["total_volume"] / 1e6, color="#2ca02c", lw=1.0)
    axes[1].fill_between(
        daily["date"],
        0,
        daily["total_volume"] / 1e6,
        where=gap.values,
        color="#d62728",
        alpha=0.25,
        label="zero/missing-volume days",
    )
    axes[1].set_title("Aggregate Daily Volume")
    axes[1].set_ylabel("Shares mn")
    axes[1].legend(loc="upper left")

    axes[2].plot(daily["date"], daily["total_weight"], color="#ff7f0e", lw=1.2)
    axes[2].axhline(100, color="black", ls="--", lw=0.8)
    axes[2].set_title("Sum of Constituent Index Weights")
    axes[2].set_ylabel("Percent")
    axes[2].set_xlabel("Date")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30)
    savefig("C3_EDA_03_market_aggregates.png")


def graph_04_returns_distribution(clean):
    r = clean["log_return"].dropna()
    r_clip = r.clip(lower=r.quantile(0.005), upper=r.quantile(0.995))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(r_clip, bins=120, stat="density", kde=True, color="#4c78a8", ax=axes[0])
    axes[0].axvline(r.mean(), color="black", ls="--", lw=1, label="mean")
    axes[0].set_title("Daily Log Return Distribution (Winsorized 0.5%-99.5%)")
    axes[0].set_xlabel("Log return")
    axes[0].legend()

    stats.probplot(r.replace([np.inf, -np.inf], np.nan).dropna(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot: Daily Log Returns vs Normal")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Sample quantiles")
    savefig("C3_EDA_04_return_distribution_and_qq.png")


def graph_05_yearly_return_dispersion(clean):
    d = clean[["date", "symbol", "log_return"]].dropna().copy()
    d["year"] = d["date"].dt.year.astype(str)
    yearly = d[d["year"].isin(sorted(d["year"].unique()))]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.boxplot(data=yearly, x="year", y="log_return", showfliers=False, color="#72b7b2", ax=ax)
    ax.set_title("Year-wise Dispersion of Daily Constituent Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Log return")
    savefig("C3_EDA_05_yearly_return_dispersion.png")


def graph_06_symbol_level_risk_return(clean):
    by_sym = clean.groupby("symbol", as_index=False).agg(
        avg_return=("log_return", "mean"),
        volatility=("log_return", "std"),
        avg_weight=("weight_pct", "mean"),
    ).dropna()
    by_sym["ann_return"] = by_sym["avg_return"] * 252
    by_sym["ann_vol"] = by_sym["volatility"] * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = np.clip(by_sym["avg_weight"] * 25, 20, 280)
    ax.scatter(by_sym["ann_vol"], by_sym["ann_return"], s=sizes, alpha=0.7, color="#1f77b4")
    for _, row in by_sym.nlargest(8, "avg_weight").iterrows():
        ax.annotate(row["symbol"], (row["ann_vol"], row["ann_return"]), fontsize=8)
    ax.set_title("Risk-Return Map by Constituent (Bubble = Avg Index Weight)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized mean return")
    savefig("C3_EDA_06_symbol_risk_return_map.png")


def graph_07_funds_nav_aum(funds):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
    for name, d in funds.items():
        axes[0].plot(d["date"], d["nav"], label=name, lw=1.2, color=colors[name])
        axes[1].plot(d["date"], d["aum"], label=name, lw=1.2, color=colors[name])
    axes[0].set_title("Index Fund NAV Trajectories")
    axes[0].set_ylabel("NAV")
    axes[0].legend()
    axes[1].set_title("Index Fund AUM Trajectories")
    axes[1].set_ylabel("AUM")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30)
    savefig("C3_EDA_07_funds_nav_aum.png")


def graph_08_fund_flow_and_corr(funds):
    monthly = []
    for fund, d in funds.items():
        m = (
            d.set_index("date")
            .resample("ME")
            .agg(nav_start=("nav", "first"), nav_end=("nav", "last"), aum=("aum", "last"))
            .reset_index()
        )
        m["aum_prev"] = m["aum"].shift(1)
        m["nav_return_m"] = m["nav_end"] / m["nav_start"] - 1
        m["fund_flow"] = m["aum"] - m["aum_prev"] * (1 + m["nav_return_m"])
        m["fund"] = fund
        monthly.append(m.dropna(subset=["fund_flow"]))
    mf = pd.concat(monthly, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NTI": "#2ca02c"}
    for fund in ["AKD", "NBP", "NTI"]:
        s = mf[mf["fund"] == fund]
        axes[0].plot(s["date"], s["fund_flow"], lw=1.2, color=colors[fund], label=fund)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Monthly Fund Flows")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Flow")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    corr = mf.pivot_table(index="date", columns="fund", values="fund_flow").corr(min_periods=8)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Correlation of Monthly Fund Flows")
    fig.autofmt_xdate(rotation=30)
    savefig("C3_EDA_08_funds_flow_and_corr.png")


def export_summary(raw, clean, funds):
    summary = []
    summary.append(
        {
            "metric": "raw_rows",
            "value": int(len(raw)),
        }
    )
    summary.append({"metric": "clean_rows", "value": int(len(clean))})
    summary.append(
        {
            "metric": "raw_volume_missing_pct",
            "value": round(float(raw["volume"].isna().mean() * 100), 2),
        }
    )
    summary.append(
        {
            "metric": "raw_volume_zero_or_missing_pct",
            "value": round(float((raw["volume"].isna() | raw["volume"].eq(0)).mean() * 100), 2),
        }
    )
    summary.append(
        {
            "metric": "clean_return_skew",
            "value": round(float(clean["log_return"].dropna().skew()), 4),
        }
    )
    summary.append(
        {
            "metric": "clean_return_excess_kurtosis",
            "value": round(float(clean["log_return"].dropna().kurtosis()), 4),
        }
    )
    for name, d in funds.items():
        summary.append({"metric": f"{name.lower()}_start_date", "value": d["date"].min().date().isoformat()})
        summary.append({"metric": f"{name.lower()}_end_date", "value": d["date"].max().date().isoformat()})
    pd.DataFrame(summary).to_csv(OUT_DIR / "C3_EDA_summary.csv", index=False)


def main():
    raw, clean, funds = load_data()
    graph_01_coverage_and_quality(raw, clean)
    graph_02_volume_gap_heatmap(raw)
    graph_03_market_aggregates(raw)
    graph_04_returns_distribution(clean)
    graph_05_yearly_return_dispersion(clean)
    graph_06_symbol_level_risk_return(clean)
    graph_07_funds_nav_aum(funds)
    graph_08_fund_flow_and_corr(funds)
    export_summary(raw, clean, funds)
    print(f"\nDone. Chapter 3 EDA figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
