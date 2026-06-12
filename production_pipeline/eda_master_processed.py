"""
EDA for processed master datasets produced by the production pipeline.

Inputs:
  - daily_master.csv
  - monthly_master.csv

Outputs (under production_pipeline/output/eda_master):
  - csvs/: summary and quality CSV files
  - figs/: EDA plots

Run:
  python production_pipeline/eda_master_processed.py
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
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ANALYSIS_DIR = BASE_DIR / "output" / "analysis"
DAILY_PATH = ANALYSIS_DIR / "daily_master.csv"
MONTHLY_PATH = ANALYSIS_DIR / "monthly_master.csv"
GOLD_PATH = RAW_DIR / "gold.csv"
GDP_PATH = RAW_DIR / "gdp.xls"
OUTPUT_DIR = BASE_DIR / "output" / "eda_master"
FIGS_DIR = OUTPUT_DIR / "figs"
CSVS_DIR = OUTPUT_DIR / "csvs"
STATIONARITY_DIR = OUTPUT_DIR / "stationarity"
STATIONARITY_FIGS_DIR = STATIONARITY_DIR / "figs"
STATIONARITY_CSVS_DIR = STATIONARITY_DIR / "csvs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
CSVS_DIR.mkdir(parents=True, exist_ok=True)
STATIONARITY_FIGS_DIR.mkdir(parents=True, exist_ok=True)
STATIONARITY_CSVS_DIR.mkdir(parents=True, exist_ok=True)

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
    pd.DataFrame([row]).to_csv(CSVS_DIR / f"{name}_overview.csv", index=False)


def save_missingness(df: pd.DataFrame, name: str) -> None:
    miss = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(3),
        }
    ).sort_values("missing_pct", ascending=False)
    miss.to_csv(CSVS_DIR / f"{name}_missingness.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=miss, x="column", y="missing_pct", color="#4C78A8")
    plt.title(f"{name}: Missingness by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{name}_missingness.png")
    plt.close()


def save_numeric_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return num
    summary = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary["skew"] = num.skew(numeric_only=True)
    summary["kurtosis"] = num.kurtosis(numeric_only=True)
    summary.to_csv(CSVS_DIR / f"{name}_numeric_summary.csv")
    return num


def plot_correlation(num_df: pd.DataFrame, name: str) -> None:
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    corr.to_csv(CSVS_DIR / f"{name}_correlation.csv")

    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title(f"{name}: Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{name}_correlation_heatmap.png")
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
    plt.savefig(FIGS_DIR / fname)
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
    plt.savefig(FIGS_DIR / f"{name}_distributions.png")
    plt.close()


def monthly_specific_checks(monthly: pd.DataFrame) -> None:
    if "date" not in monthly.columns:
        return
    out = monthly.copy()
    out["year"] = out["date"].dt.year
    if "flow_spike_sector" in out.columns:
        spike_summary = out["flow_spike_sector"].value_counts(dropna=False).rename_axis("flow_spike_sector").reset_index(name="count")
        spike_summary.to_csv(CSVS_DIR / "monthly_flow_spike_counts.csv", index=False)

    if "total_fund_flow" in out.columns:
        yearly_flow = out.groupby("year", dropna=True)["total_fund_flow"].agg(["count", "mean", "median", "std", "min", "max"])
        yearly_flow.to_csv(CSVS_DIR / "monthly_total_fund_flow_by_year.csv")

        plt.figure(figsize=(10, 4))
        sns.boxplot(data=out, x="year", y="total_fund_flow", color="#F58518")
        plt.title("Monthly Total Fund Flow by Year")
        plt.xlabel("Year")
        plt.ylabel("Total Fund Flow")
        plt.tight_layout()
        plt.savefig(FIGS_DIR / "monthly_total_fund_flow_boxplot_by_year.png")
        plt.close()


def build_enriched_masters(daily: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_merged = daily.copy()
    monthly_merged = monthly.copy()

    if GOLD_PATH.exists():
        gold = load_gold(GOLD_PATH)
        if "date" in daily_merged.columns:
            daily_merged = daily_merged.merge(gold[["date", "gold_price", "gold_log_return", "gold_volume"]], on="date", how="left")

        if "date" in monthly_merged.columns:
            monthly_gold = (
                gold.set_index("date")
                .resample("ME")
                .agg(
                    gold_price_end=("gold_price", "last"),
                    gold_return_monthly=("gold_price", lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else np.nan),
                    gold_volume_end=("gold_volume", "last"),
                )
                .reset_index()
            )
            monthly_merged = monthly_merged.merge(monthly_gold.rename(columns={"date": "date"}), on="date", how="left")

    if GDP_PATH.exists():
        gdp = load_gdp(GDP_PATH)
        if "date" in monthly_merged.columns:
            gdp_monthly = gdp.set_index("date").resample("ME").ffill().reset_index()
            monthly_merged = monthly_merged.sort_values("date").merge(gdp_monthly, on="date", how="left")
            monthly_merged[["gdp_usd", "gdp_yoy"]] = monthly_merged[["gdp_usd", "gdp_yoy"]].ffill()

    return daily_merged, monthly_merged


def stationarity_tests(df: pd.DataFrame, name: str, min_obs: int = 24) -> None:
    num = df.select_dtypes(include=[np.number]).copy()
    results = []
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < min_obs or s.nunique() < 3:
            continue

        adf_p = np.nan
        kpss_p = np.nan
        try:
            adf_p = adfuller(s, autolag="AIC")[1]
        except Exception:
            pass
        try:
            kpss_p = kpss(s, regression="c", nlags="auto")[1]
        except Exception:
            pass

        adf_stationary = bool(adf_p < 0.05) if pd.notna(adf_p) else False
        kpss_stationary = bool(kpss_p > 0.05) if pd.notna(kpss_p) else False
        overall_stationary = adf_stationary and kpss_stationary
        results.append(
            {
                "dataset": name,
                "variable": col,
                "n_obs": len(s),
                "adf_pvalue": adf_p,
                "kpss_pvalue": kpss_p,
                "adf_stationary_5pct": adf_stationary,
                "kpss_stationary_5pct": kpss_stationary,
                "overall_stationary_5pct": overall_stationary,
            }
        )

    out = pd.DataFrame(results).sort_values("variable")
    out.to_csv(STATIONARITY_CSVS_DIR / f"{name}_stationarity_tests.csv", index=False)

    if out.empty:
        return

    pvals = out.set_index("variable")[["adf_pvalue", "kpss_pvalue"]]
    plt.figure(figsize=(10, max(4, len(pvals) * 0.25)))
    sns.heatmap(pvals, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0, vmax=0.1, linewidths=0.2)
    plt.title(f"{name}: Stationarity Test P-Values (ADF and KPSS)")
    plt.tight_layout()
    plt.savefig(STATIONARITY_FIGS_DIR / f"{name}_stationarity_pvalues_heatmap.png")
    plt.close()

    counts = (
        out["overall_stationary_5pct"]
        .map({True: "Stationary", False: "Non-stationary"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    counts.to_csv(STATIONARITY_CSVS_DIR / f"{name}_stationarity_class_counts.csv", index=False)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=counts, x="class", y="count", palette=["#54A24B", "#E45756"])
    plt.title(f"{name}: Stationarity Classification (5% level)")
    plt.xlabel("")
    plt.ylabel("Variable Count")
    plt.tight_layout()
    plt.savefig(STATIONARITY_FIGS_DIR / f"{name}_stationarity_classification.png")
    plt.close()


def main() -> None:
    print("Loading processed masters...")
    daily = load_master(DAILY_PATH)
    monthly = load_master(MONTHLY_PATH)

    print("Saving overview and quality summaries...")
    daily_all, monthly_all = build_enriched_masters(daily, monthly)

    save_overview(daily_all, "daily_master_all")
    save_overview(monthly_all, "monthly_master_all")
    save_missingness(daily_all, "daily_master_all")
    save_missingness(monthly_all, "monthly_master_all")

    daily_num = save_numeric_summary(daily_all, "daily_master_all")
    monthly_num = save_numeric_summary(monthly_all, "monthly_master_all")
    plot_correlation(daily_num, "daily_master_all")
    plot_correlation(monthly_num, "monthly_master_all")
    plot_distributions(daily_num, "daily_master_all")
    plot_distributions(monthly_num, "monthly_master_all")

    plot_time_series(
        daily_all,
        [
            "idx_total_volume",
            "idx_ff_mcap_total",
            "idx_log_return",
            "idx_rolling_vol_30d",
            "oil_price",
            "usdpkr",
            "cpi_yoy",
            "gold_price",
            "gold_log_return",
        ],
        "daily_master_all",
        "daily_master_all_key_timeseries.png",
    )
    plot_time_series(
        monthly_all,
        [
            "total_fund_flow",
            "flow_pct_sector",
            "idx_return_monthly",
            "idx_vol_monthly",
            "oil_price_end",
            "usdpkr_end",
            "cpi_yoy_end",
            "gold_price_end",
            "gold_return_monthly",
            "gdp_usd",
            "gdp_yoy",
        ],
        "monthly_master_all",
        "monthly_master_all_key_timeseries.png",
    )
    monthly_specific_checks(monthly_all)
    stationarity_tests(daily_all, "daily_master_all")
    stationarity_tests(monthly_all, "monthly_master_all")

    print(f"EDA complete. Outputs saved to: {OUTPUT_DIR}")
    print(f"Figures: {FIGS_DIR}")
    print(f"CSVs: {CSVS_DIR}")


if __name__ == "__main__":
    main()
