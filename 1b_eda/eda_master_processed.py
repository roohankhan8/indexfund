"""
EDA for processed master datasets in 1b_eda.

Inputs:
  - daily_master.csv
  - monthly_master.csv

Outputs (under 1b_eda/output):
  - summary CSV files
  - quality checks
  - EDA plots

Run:
  python 1b_eda/eda_master_processed.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DAILY_PATH = BASE_DIR / "daily_master.csv"
MONTHLY_PATH = BASE_DIR / "monthly_master.csv"
GOLD_PATH = BASE_DIR / "gold.csv"
GDP_PATH = BASE_DIR / "gdp.xls"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            bool_map = {"true": True, "false": False}
            maybe_bool = cleaned.str.lower().map(bool_map)
            if maybe_bool.notna().sum() == cleaned.notna().sum():
                df[col] = maybe_bool
                continue
            df[col] = pd.to_numeric(cleaned, errors="ignore")
    return df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df


def _parse_human_volume(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper().replace(",", "")
    if s.endswith("K"):
        return float(s[:-1]) * 1_000
    if s.endswith("M"):
        return float(s[:-1]) * 1_000_000
    if s.endswith("B"):
        return float(s[:-1]) * 1_000_000_000
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "Price": "gold_price", "Vol.": "gold_volume", "Change %": "gold_change_pct"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["gold_price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "gold_change_pct" in df.columns:
        df["gold_change_pct"] = pd.to_numeric(df["gold_change_pct"].astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0
    if "gold_volume" in df.columns:
        df["gold_volume"] = df["gold_volume"].map(_parse_human_volume)
    df = df.sort_values("date").reset_index(drop=True)
    df["gold_log_return"] = np.log(df["gold_price"] / df["gold_price"].shift(1))
    return df


def load_gdp(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "gdp_usd"])
    pak = raw.loc[raw["Country Code"].astype(str).str.upper() == "PAK"].copy()
    if pak.empty:
        pak = raw.iloc[[0]].copy()
    year_cols = [c for c in pak.columns if str(c).isdigit()]
    long = pak.melt(value_vars=year_cols, var_name="year", value_name="gdp_usd")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["gdp_usd"] = pd.to_numeric(long["gdp_usd"], errors="coerce")
    long = long.dropna(subset=["year"]).sort_values("year")
    long["date"] = pd.to_datetime(long["year"].astype(int).astype(str) + "-12-31", errors="coerce")
    long["gdp_yoy"] = long["gdp_usd"].pct_change()
    return long[["date", "gdp_usd", "gdp_yoy"]].reset_index(drop=True)


def save_overview(df: pd.DataFrame, name: str) -> None:
    row = {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "date_min": df["date"].min() if "date" in df.columns else pd.NaT,
        "date_max": df["date"].max() if "date" in df.columns else pd.NaT,
    }
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / f"{name}_overview.csv", index=False)


def save_missingness(df: pd.DataFrame, name: str) -> None:
    miss = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(3),
        }
    ).sort_values("missing_pct", ascending=False)
    miss.to_csv(OUTPUT_DIR / f"{name}_missingness.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=miss, x="column", y="missing_pct", color="#4C78A8")
    plt.title(f"{name}: Missingness by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_missingness.png")
    plt.close()


def save_numeric_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return num
    summary = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary["skew"] = num.skew(numeric_only=True)
    summary["kurtosis"] = num.kurtosis(numeric_only=True)
    summary.to_csv(OUTPUT_DIR / f"{name}_numeric_summary.csv")
    return num


def plot_correlation(num_df: pd.DataFrame, name: str) -> None:
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / f"{name}_correlation.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title(f"{name}: Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_correlation_heatmap.png")
    plt.close()


def plot_time_series(df: pd.DataFrame, cols: list[str], name: str, fname: str) -> None:
    available = [c for c in cols if c in df.columns]
    if "date" not in df.columns or not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.6 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, col in zip(axes, available):
        ax.plot(df["date"], df[col], linewidth=1.2)
        ax.set_title(col)
    fig.suptitle(f"{name}: Key Time Series", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()


def plot_distributions(num_df: pd.DataFrame, name: str, max_cols: int = 8) -> None:
    if num_df.empty:
        return
    cols = list(num_df.columns)[:max_cols]
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sns.histplot(num_df[col].dropna(), bins=40, kde=True, ax=axes[i], color="#72B7B2")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name}: Numeric Distributions (first {len(cols)} cols)", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_distributions.png")
    plt.close()


def monthly_specific_checks(monthly: pd.DataFrame) -> None:
    if "date" not in monthly.columns:
        return
    out = monthly.copy()
    out["year"] = out["date"].dt.year
    if "flow_spike_sector" in out.columns:
        spike_summary = out["flow_spike_sector"].value_counts(dropna=False).rename_axis("flow_spike_sector").reset_index(name="count")
        spike_summary.to_csv(OUTPUT_DIR / "monthly_flow_spike_counts.csv", index=False)

    if "total_fund_flow" in out.columns:
        yearly_flow = out.groupby("year", dropna=True)["total_fund_flow"].agg(["count", "mean", "median", "std", "min", "max"])
        yearly_flow.to_csv(OUTPUT_DIR / "monthly_total_fund_flow_by_year.csv")

        plt.figure(figsize=(10, 4))
        sns.boxplot(data=out, x="year", y="total_fund_flow", color="#F58518")
        plt.title("Monthly Total Fund Flow by Year")
        plt.xlabel("Year")
        plt.ylabel("Total Fund Flow")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "monthly_total_fund_flow_boxplot_by_year.png")
        plt.close()


def external_macro_eda(daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    if GOLD_PATH.exists():
        gold = load_gold(GOLD_PATH)
        save_overview(gold, "gold")
        save_missingness(gold, "gold")
        gold_num = save_numeric_summary(gold, "gold")
        plot_correlation(gold_num, "gold")

        plot_time_series(gold, ["gold_price", "gold_volume", "gold_log_return"], "gold", "gold_key_timeseries.png")

        if "date" in daily.columns:
            daily_gold = daily.merge(gold[["date", "gold_price", "gold_log_return"]], on="date", how="left")
            save_missingness(daily_gold, "daily_master_with_gold")
            daily_gold_num = daily_gold.select_dtypes(include=[np.number])
            plot_correlation(daily_gold_num, "daily_master_with_gold")

        if "date" in monthly.columns:
            monthly_gold = (
                gold.set_index("date")
                .resample("ME")
                .agg(gold_price_end=("gold_price", "last"), gold_return_monthly=("gold_price", lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else np.nan))
                .reset_index()
            )
            merged = monthly.merge(monthly_gold.rename(columns={"date": "date"}), on="date", how="left")
            save_missingness(merged, "monthly_master_with_gold")
            plot_correlation(merged.select_dtypes(include=[np.number]), "monthly_master_with_gold")

    if GDP_PATH.exists():
        gdp = load_gdp(GDP_PATH)
        save_overview(gdp, "gdp")
        save_missingness(gdp, "gdp")
        gdp_num = save_numeric_summary(gdp, "gdp")
        plot_correlation(gdp_num, "gdp")
        plot_time_series(gdp, ["gdp_usd", "gdp_yoy"], "gdp", "gdp_key_timeseries.png")

        if "date" in monthly.columns:
            monthly_gdp = monthly.copy().sort_values("date")
            gdp_monthly = gdp.set_index("date").resample("ME").ffill().reset_index()
            merged = monthly_gdp.merge(gdp_monthly, on="date", how="left")
            merged[["gdp_usd", "gdp_yoy"]] = merged[["gdp_usd", "gdp_yoy"]].ffill()
            save_missingness(merged, "monthly_master_with_gdp")
            plot_correlation(merged.select_dtypes(include=[np.number]), "monthly_master_with_gdp")


def main() -> None:
    print("Loading processed masters...")
    daily = load_master(DAILY_PATH)
    monthly = load_master(MONTHLY_PATH)

    print("Saving overview and quality summaries...")
    save_overview(daily, "daily_master")
    save_overview(monthly, "monthly_master")
    save_missingness(daily, "daily_master")
    save_missingness(monthly, "monthly_master")

    daily_num = save_numeric_summary(daily, "daily_master")
    monthly_num = save_numeric_summary(monthly, "monthly_master")
    plot_correlation(daily_num, "daily_master")
    plot_correlation(monthly_num, "monthly_master")
    plot_distributions(daily_num, "daily_master")
    plot_distributions(monthly_num, "monthly_master")

    plot_time_series(
        daily,
        ["idx_total_volume", "idx_ff_mcap_total", "idx_log_return", "idx_rolling_vol_30d", "oil_price", "usdpkr", "cpi_yoy"],
        "daily_master",
        "daily_master_key_timeseries.png",
    )
    plot_time_series(
        monthly,
        ["total_fund_flow", "flow_pct_sector", "idx_return_monthly", "idx_vol_monthly", "oil_price_end", "usdpkr_end", "cpi_yoy_end"],
        "monthly_master",
        "monthly_master_key_timeseries.png",
    )
    monthly_specific_checks(monthly)
    external_macro_eda(daily, monthly)

    print(f"EDA complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
