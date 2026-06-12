"""
Transform production-pipeline data into stationarity-friendly modeling series and report results.

Outputs (docs/report_workspace/chapter-03-methodology/images):
  - C3_stationarity_before_after.csv
  - C3_modeling_daily_transformed.csv
  - C3_modeling_monthly_transformed.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from arch.unitroot import PhillipsPerron
    HAS_PP = True
except Exception:
    HAS_PP = False


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ANALYSIS_DIR = BASE_DIR / "output" / "analysis"
OUT = BASE_DIR.parent / "docs" / "report_workspace" / "chapter-03-methodology" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def test_stationarity(series: pd.Series):
    x = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < 25:
        return {
            "n_obs": len(x),
            "adf_p": np.nan,
            "pp_p": np.nan,
            "kpss_p": np.nan,
            "adf_stationary": np.nan,
            "pp_stationary": np.nan,
            "kpss_stationary": np.nan,
            "overall": "Insufficient",
        }

    try:
        adf_p = adfuller(x, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan

    if HAS_PP:
        try:
            pp_p = float(PhillipsPerron(x).pvalue)
        except Exception:
            pp_p = np.nan
    else:
        pp_p = np.nan

    try:
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan

    adf_s = pd.notna(adf_p) and adf_p < 0.05
    pp_s = pd.notna(pp_p) and pp_p < 0.05
    kpss_s = pd.notna(kpss_p) and kpss_p > 0.05

    if adf_s and kpss_s:
        overall = "Stationary"
    elif (not adf_s) and (not kpss_s):
        overall = "Non-stationary"
    else:
        overall = "Mixed"

    return {
        "n_obs": len(x),
        "adf_p": adf_p,
        "pp_p": pp_p,
        "kpss_p": kpss_p,
        "adf_stationary": adf_s,
        "pp_stationary": pp_s if pd.notna(pp_p) else np.nan,
        "kpss_stationary": kpss_s,
        "overall": overall,
    }


def load_data():
    raw = pd.read_csv(RAW_DIR / "kse30_daily_data.csv").rename(
        columns={
            "Date": "date",
            "PRICE": "price",
            "IDX WT %": "weight_pct",
            "FF BASED MCAP": "ff_mcap",
            "VOLUME": "volume",
            "SYMBOL": "symbol",
        }
    )
    raw["date"] = pd.to_datetime(raw["date"])
    for c in ["price", "weight_pct", "ff_mcap", "volume"]:
        if raw[c].dtype == "object":
            raw[c] = raw[c].str.replace(",", "", regex=False)
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    clean = pd.read_csv(ANALYSIS_DIR / "kse30_stocks_clean.csv")
    clean["date"] = pd.to_datetime(clean["date"])
    for c in ["log_return", "price", "ff_mcap", "volume", "weight_pct"]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    funds = {}
    for fund in ["AKD", "NBP", "NTI"]:
        df = pd.read_excel(RAW_DIR / "funds_data.xlsx", sheet_name=fund).rename(
            columns={"DATE": "date", "NAV": "nav", "AUM": "aum"}
        )
        df["date"] = pd.to_datetime(df["date"])
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["aum"] = pd.to_numeric(df["aum"], errors="coerce")
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
        funds[fund] = df

    return raw, clean, funds


def build_transformed(raw, clean, funds):
    agg = (
        raw.groupby("date", as_index=False)
        .agg(
            kse_total_ff_mcap=("ff_mcap", "sum"),
            kse_total_volume=("volume", "sum"),
            kse_avg_price=("price", "mean"),
            kse_weight_sum=("weight_pct", "sum"),
        )
        .sort_values("date")
    )

    daily = pd.DataFrame({"date": agg["date"]})
    daily["kse_ff_mcap_logdiff"] = np.log(agg["kse_total_ff_mcap"] / agg["kse_total_ff_mcap"].shift(1))
    daily["kse_avg_price_logdiff"] = np.log(agg["kse_avg_price"] / agg["kse_avg_price"].shift(1))

    # Volume has zero stretches; use log1p first-difference to reduce scale and keep zero-safe transform.
    daily["kse_volume_log1p_diff"] = np.log1p(agg["kse_total_volume"]).diff()

    # Weight sum is already bounded around 100; center it.
    daily["kse_weight_sum_centered"] = agg["kse_weight_sum"] - 100.0

    # Representative stationary cross-sectional return
    daily_ret = clean.groupby("date")["log_return"].mean().rename("kse_constituent_mean_log_return").reset_index()
    daily = daily.merge(daily_ret, on="date", how="left")

    monthly_frames = []
    for fund, df in funds.items():
        m = (
            df.set_index("date")
            .resample("ME")
            .agg(nav_start=("nav", "first"), nav_end=("nav", "last"), aum=("aum", "last"))
            .reset_index()
        )
        m["aum_prev"] = m["aum"].shift(1)
        m["nav_return_m"] = m["nav_end"] / m["nav_start"] - 1
        m["fund_flow"] = m["aum"] - m["aum_prev"] * (1 + m["nav_return_m"])
        m["fund_flow_pct"] = m["fund_flow"] / m["aum_prev"]
        m[f"{fund.lower()}_nav_logdiff"] = np.log(m["nav_end"] / m["nav_end"].shift(1))
        m[f"{fund.lower()}_aum_logdiff"] = np.log(m["aum"] / m["aum"].shift(1))
        m = m.rename(
            columns={
                "date": "month",
                "fund_flow": f"{fund.lower()}_fund_flow",
                "fund_flow_pct": f"{fund.lower()}_fund_flow_pct",
                "nav_return_m": f"{fund.lower()}_nav_return_m",
            }
        )
        monthly_frames.append(
            m[
                [
                    "month",
                    f"{fund.lower()}_fund_flow",
                    f"{fund.lower()}_fund_flow_pct",
                    f"{fund.lower()}_nav_return_m",
                    f"{fund.lower()}_nav_logdiff",
                    f"{fund.lower()}_aum_logdiff",
                ]
            ]
        )

    monthly = monthly_frames[0]
    for frame in monthly_frames[1:]:
        monthly = monthly.merge(frame, on="month", how="outer")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["total_fund_flow"] = monthly[[c for c in monthly.columns if c.endswith("_fund_flow")]].sum(axis=1, min_count=1)
    monthly["total_fund_flow_pct_mean"] = monthly[[c for c in monthly.columns if c.endswith("_fund_flow_pct")]].mean(axis=1)

    return agg, daily, monthly


def compare_stationarity(agg, daily, funds, monthly):
    tests = []

    before_series = {
        "kse_total_ff_mcap_level": agg.set_index("date")["kse_total_ff_mcap"],
        "kse_total_volume_level": agg.set_index("date")["kse_total_volume"],
        "kse_avg_price_level": agg.set_index("date")["kse_avg_price"],
        "kse_weight_sum_level": agg.set_index("date")["kse_weight_sum"],
    }
    for fund, df in funds.items():
        before_series[f"{fund.lower()}_nav_level"] = df.set_index("date")["nav"]
    for name, s in before_series.items():
        r = test_stationarity(s)
        r.update({"series": name, "stage": "before"})
        tests.append(r)

    after_series = {
        "kse_ff_mcap_logdiff": daily.set_index("date")["kse_ff_mcap_logdiff"],
        "kse_volume_log1p_diff": daily.set_index("date")["kse_volume_log1p_diff"],
        "kse_avg_price_logdiff": daily.set_index("date")["kse_avg_price_logdiff"],
        "kse_weight_sum_centered": daily.set_index("date")["kse_weight_sum_centered"],
        "kse_constituent_mean_log_return": daily.set_index("date")["kse_constituent_mean_log_return"],
        "total_fund_flow": monthly.set_index("month")["total_fund_flow"],
        "total_fund_flow_pct_mean": monthly.set_index("month")["total_fund_flow_pct_mean"],
    }
    for fund in ["akd", "nbp", "nti"]:
        for col in [f"{fund}_fund_flow", f"{fund}_fund_flow_pct", f"{fund}_nav_return_m", f"{fund}_nav_logdiff", f"{fund}_aum_logdiff"]:
            after_series[col] = monthly.set_index("month")[col]

    for name, s in after_series.items():
        r = test_stationarity(s)
        r.update({"series": name, "stage": "after"})
        tests.append(r)

    return pd.DataFrame(tests)[
        [
            "stage",
            "series",
            "n_obs",
            "adf_p",
            "pp_p",
            "kpss_p",
            "adf_stationary",
            "pp_stationary",
            "kpss_stationary",
            "overall",
        ]
    ].sort_values(["stage", "series"])


def main():
    raw, clean, funds = load_data()
    agg, daily, monthly = build_transformed(raw, clean, funds)
    stationarity = compare_stationarity(agg, daily, funds, monthly)

    daily.to_csv(OUT / "C3_modeling_daily_transformed.csv", index=False)
    monthly.to_csv(OUT / "C3_modeling_monthly_transformed.csv", index=False)
    stationarity.to_csv(OUT / "C3_stationarity_before_after.csv", index=False)

    print(f"PP available: {HAS_PP}")
    print("Saved transformed datasets and stationarity table:")
    print(f"  {OUT / 'C3_modeling_daily_transformed.csv'}")
    print(f"  {OUT / 'C3_modeling_monthly_transformed.csv'}")
    print(f"  {OUT / 'C3_stationarity_before_after.csv'}")

    summary = stationarity.groupby(["stage", "overall"]).size().rename("count").reset_index()
    print("\nStationarity summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
