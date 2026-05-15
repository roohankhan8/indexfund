from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
TABLE_DIR = OUT_DIR / "tables"
METRIC_DIR = OUT_DIR / "metrics"
FIG_DIR = OUT_DIR / "figures"
FUNDS = ["AKD", "NBP", "NTI"]


# Flow_t = AUM_t - AUM_{t-1} * (NAV_t / NAV_{t-1})
def compute_flow(df: pd.DataFrame) -> pd.Series:
    nav_lag = df["NAV"].shift(1)
    aum_lag = df["AUM"].shift(1)
    return df["AUM"] - aum_lag * (df["NAV"] / nav_lag)


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, METRIC_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_funds_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(DATA_DIR / "funds_data.xlsx")
    monthly_blocks = []
    daily_blocks = []

    for fund in FUNDS:
        df = pd.read_excel(xls, sheet_name=fund)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[["DATE", "NAV", "AUM"]].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").dropna()
        df["FLOW"] = compute_flow(df)
        df["FUND"] = fund
        daily_blocks.append(df.copy())

        m = (
            df.set_index("DATE")[["NAV", "AUM", "FLOW"]]
            .resample("ME")
            .last()
            .dropna()
            .rename(columns={"NAV": f"NAV_{fund}", "AUM": f"AUM_{fund}", "FLOW": f"FLOW_{fund}"})
        )
        m[f"RET_{fund}"] = m[f"NAV_{fund}"].pct_change()
        monthly_blocks.append(m)

    monthly = pd.concat(monthly_blocks, axis=1, sort=False).sort_index().dropna()
    monthly["FLOW_TOTAL"] = monthly[[f"FLOW_{f}" for f in FUNDS]].sum(axis=1)
    daily_all = pd.concat(daily_blocks, ignore_index=True)
    return monthly, daily_all


def load_market_monthly() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "kse30_daily_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["SYMBOL", "Date"])

    df["symbol_ret"] = df.groupby("SYMBOL")["PRICE"].pct_change()
    df["w"] = pd.to_numeric(df["IDX WT %"], errors="coerce")
    df["w"] = df["w"].fillna(0)

    daily = (
        df.groupby("Date")
        .apply(
            lambda g: pd.Series(
                {
                    "mkt_ret": np.average(g["symbol_ret"].fillna(0), weights=(g["w"] + 1e-9)),
                    "mkt_vol": g["symbol_ret"].std(ddof=0),
                    "mkt_turnover": g["VOLUME"].mean(),
                }
            )
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.set_index("Date").sort_index()

    monthly = pd.DataFrame(index=daily.resample("ME").last().index)
    monthly["mkt_ret_mean"] = daily["mkt_ret"].resample("ME").mean()
    monthly["mkt_vol_mean"] = daily["mkt_vol"].resample("ME").mean()
    monthly["mkt_vol_std"] = daily["mkt_ret"].resample("ME").std()
    monthly["mkt_turnover_mean"] = daily["mkt_turnover"].resample("ME").mean()
    return monthly.dropna()


def load_macro_monthly() -> pd.DataFrame:
    xls = pd.ExcelFile(DATA_DIR / "macro_data.xlsx")
    oil = pd.read_excel(xls, sheet_name="OIL")
    ir = pd.read_excel(xls, sheet_name="IR")
    usd = pd.read_excel(xls, sheet_name="USD")

    oil["DATE"] = pd.to_datetime(oil["DATE"])
    ir["DATE"] = pd.to_datetime(ir["DATE"])
    usd["DATE"] = pd.to_datetime(usd["DATE"])

    oil_m = oil.set_index("DATE")["PRICE"].resample("ME").last().rename("oil")
    ir_m = ir.set_index("DATE")["RATE"].resample("ME").last().rename("ir")
    usd_m = usd.set_index("DATE")["USD"].resample("ME").last().rename("usd")

    cpi = pd.read_csv(DATA_DIR / "cpi.csv")
    cpi = cpi.rename(columns={cpi.columns[0]: "period", cpi.columns[1]: "cpi_yoy"})
    cpi = cpi[cpi["period"].str.lower() != "period"].copy()
    def parse_cpi_period(s: str) -> pd.Timestamp | pd.NaT:
        for fmt in ("%b-%y", "%B-%y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except (TypeError, ValueError):
                continue
        return pd.NaT

    cpi["DATE"] = cpi["period"].map(parse_cpi_period) + pd.offsets.MonthEnd(0)
    cpi["cpi_yoy"] = pd.to_numeric(cpi["cpi_yoy"], errors="coerce")
    cpi_m = cpi.dropna(subset=["DATE"]).groupby("DATE", as_index=True)["cpi_yoy"].last()

    start = min(oil_m.index.min(), ir_m.index.min(), usd_m.index.min(), cpi_m.index.min())
    end = max(oil_m.index.max(), ir_m.index.max(), usd_m.index.max(), cpi_m.index.max())
    monthly_index = pd.date_range(start=start, end=end, freq="ME")

    macro = pd.concat([oil_m, ir_m, usd_m, cpi_m], axis=1, sort=False).reindex(monthly_index).sort_index()
    macro[["oil", "ir", "usd"]] = macro[["oil", "ir", "usd"]].ffill()
    macro["oil_ret"] = macro["oil"].pct_change()
    macro["usd_ret"] = macro["usd"].pct_change()
    macro["ir_chg"] = macro["ir"].diff()
    return macro.dropna()


def build_model_frame(monthly_funds: pd.DataFrame, market: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = monthly_funds.join(market, how="inner").join(macro, how="inner")

    base_predictors = [
        "mkt_ret_mean",
        "mkt_vol_mean",
        "mkt_vol_std",
        "mkt_turnover_mean",
        "oil_ret",
        "usd_ret",
        "ir_chg",
        "cpi_yoy",
        "FLOW_TOTAL",
    ] + [f"FLOW_{f}" for f in FUNDS]

    for col in base_predictors:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for fund in FUNDS:
        df[f"target_flow_{fund}"] = df[f"FLOW_{fund}"].shift(-1)
        df[f"target_ret_{fund}"] = df[f"RET_{fund}"].shift(-1)

    return df.dropna().copy()


def chrono_split(df: pd.DataFrame, test_ratio: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(6, int(len(df) * test_ratio))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test


def fit_predict_by_fund(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    pred_df = pd.DataFrame(index=test.index)
    metrics = {}

    for fund in FUNDS:
        y_train = train[f"target_flow_{fund}"]
        y_test = test[f"target_flow_{fund}"]

        reg = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2)
        reg.fit(train[feature_cols], y_train)
        pred_flow = reg.predict(test[feature_cols])

        cls = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=2)
        cls.fit(train[feature_cols], (y_train > 0).astype(int))
        pred_dir = cls.predict(test[feature_cols])

        actual_dir = (y_test > 0).astype(int)
        pred_df[f"actual_flow_{fund}"] = y_test.values
        pred_df[f"pred_flow_{fund}"] = pred_flow
        pred_df[f"actual_dir_{fund}"] = actual_dir.values
        pred_df[f"pred_dir_{fund}"] = pred_dir
        pred_df[f"realized_ret_{fund}"] = test[f"target_ret_{fund}"].values

        metrics[fund] = {
            "flow_rmse": float(np.sqrt(mean_squared_error(y_test, pred_flow))),
            "flow_mae": float(mean_absolute_error(y_test, pred_flow)),
            "flow_r2": float(r2_score(y_test, pred_flow)),
            "dir_accuracy": float(accuracy_score(actual_dir, pred_dir)),
            "dir_f1": float(f1_score(actual_dir, pred_dir, zero_division=0)),
        }

    pred_df["month"] = pred_df.index
    return pred_df, metrics


def build_strategy(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    for fund in FUNDS:
        out[f"score_{fund}"] = np.maximum(out[f"pred_flow_{fund}"], 0.0)

    score_cols = [f"score_{f}" for f in FUNDS]
    score_sum = out[score_cols].sum(axis=1)

    for fund in FUNDS:
        out[f"w_tilt_{fund}"] = np.where(score_sum > 0, out[f"score_{fund}"] / score_sum, 1.0 / len(FUNDS))
        out[f"w_eq_{fund}"] = 1.0 / len(FUNDS)

    out["ret_tilt"] = 0.0
    out["ret_eq"] = 0.0
    for fund in FUNDS:
        out["ret_tilt"] += out[f"w_tilt_{fund}"] * out[f"realized_ret_{fund}"]
        out["ret_eq"] += out[f"w_eq_{fund}"] * out[f"realized_ret_{fund}"]

    out["cum_tilt"] = (1 + out["ret_tilt"].fillna(0)).cumprod()
    out["cum_eq"] = (1 + out["ret_eq"].fillna(0)).cumprod()
    return out


def save_outputs(model_df: pd.DataFrame, pred_df: pd.DataFrame, strat_df: pd.DataFrame, metrics: dict) -> None:
    model_df.to_csv(TABLE_DIR / "model_frame_monthly.csv", index=True)
    pred_df.to_csv(TABLE_DIR / "test_predictions.csv", index=False)
    strat_df.to_csv(TABLE_DIR / "strategy_backtest.csv", index=False)

    summary = {
        "months_in_model_frame": int(len(model_df)),
        "months_in_test": int(len(pred_df)),
        "fund_metrics": metrics,
        "strategy": {
            "total_return_tilt": float(strat_df["cum_tilt"].iloc[-1] - 1),
            "total_return_equal_weight": float(strat_df["cum_eq"].iloc[-1] - 1),
            "avg_monthly_return_tilt": float(strat_df["ret_tilt"].mean()),
            "avg_monthly_return_equal_weight": float(strat_df["ret_eq"].mean()),
        },
    }

    (METRIC_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 5))
    plt.plot(strat_df["month"], strat_df["cum_tilt"], label="Predicted-Flow Tilt")
    plt.plot(strat_df["month"], strat_df["cum_eq"], label="Equal Weight", linestyle="--")
    plt.title("Out-of-Sample Cumulative Return: Tilt vs Equal Weight")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_cumulative_return.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()

    monthly_funds, _ = load_funds_monthly()
    market = load_market_monthly()
    macro = load_macro_monthly()
    model_df = build_model_frame(monthly_funds, market, macro)

    feature_cols = [c for c in model_df.columns if c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_lag3")]
    train, test = chrono_split(model_df, test_ratio=0.25)

    pred_df, metrics = fit_predict_by_fund(train, test, feature_cols)
    strat_df = build_strategy(pred_df)
    save_outputs(model_df, pred_df, strat_df, metrics)

    print("Pipeline completed.")
    print(f"Model months: {len(model_df)}, Test months: {len(test)}")
    print(f"Metrics: {METRIC_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
