from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"

CUTOFF_DATE = pd.Timestamp("2025-12-31")
TARGET_NEXT_REBALANCE = pd.Timestamp("2026-03-31")  # Proxy for Mar 15, 2026 rebalance cycle
FUNDS = ["AKD", "NBP", "NTI"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def pick_first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def compute_flow(df: pd.DataFrame) -> pd.Series:
    nav_lag = df["NAV"].shift(1)
    aum_lag = df["AUM"].shift(1)
    return df["AUM"] - aum_lag * (df["NAV"] / nav_lag)


def load_fund_features() -> pd.DataFrame:
    xls_path = pick_first_existing([
        REPO / "8_last_model" / "data" / "funds_data.xlsx",
        REPO / "5_claude_pipeline" / "funds_data.xlsx",
    ])
    xls = pd.ExcelFile(xls_path)

    blocks = []
    for fund in FUNDS:
        df = pd.read_excel(xls, sheet_name=fund)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[["DATE", "NAV", "AUM"]].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").dropna()
        df["FLOW"] = compute_flow(df)

        m = df.set_index("DATE")[["NAV", "AUM", "FLOW"]].resample("ME").last()
        m = m.rename(columns={"NAV": f"NAV_{fund}", "AUM": f"AUM_{fund}", "FLOW": f"FLOW_{fund}"})
        blocks.append(m)

    out = pd.concat(blocks, axis=1, sort=False).sort_index()
    out["NAV_TOTAL"] = out[[f"NAV_{f}" for f in FUNDS]].sum(axis=1)
    out["AUM_TOTAL"] = out[[f"AUM_{f}" for f in FUNDS]].sum(axis=1)
    out["FLOW_TOTAL"] = out[[f"FLOW_{f}" for f in FUNDS]].sum(axis=1)

    for col in out.columns:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag2"] = out[col].shift(2)

    out["aum_nav_ratio"] = out["AUM_TOTAL"] / out["NAV_TOTAL"].replace(0, np.nan)
    out["flow_to_aum"] = out["FLOW_TOTAL"] / out["AUM_TOTAL"].replace(0, np.nan)
    return out


def load_inflation_features() -> pd.DataFrame:
    p = pick_first_existing([
        REPO / "8_last_model" / "data" / "inflation.xlsx",
    ])
    df = pd.read_excel(p, sheet_name=0)
    df = df[df["Country Code"].astype(str).str.upper() == "PAK"].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "inflation_annual", "inflation_lag1", "inflation_lag2", "inflation_chg"])

    row = df.iloc[0]
    year_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and str(c).isdigit())]

    vals = []
    for c in year_cols:
        vals.append((int(c), pd.to_numeric(row[c], errors="coerce")))

    out = pd.DataFrame(vals, columns=["year", "inflation_annual"]).sort_values("year")
    out["inflation_lag1"] = out["inflation_annual"].shift(1)
    out["inflation_lag2"] = out["inflation_annual"].shift(2)
    out["inflation_chg"] = out["inflation_annual"].diff()
    return out


def build_rebalance_panel() -> pd.DataFrame:
    csv_path = pick_first_existing([
        REPO / "6_cursor_model" / "kse30_stocks_clean.csv",
        REPO / "5_claude_pipeline" / "kse30_daily_data.csv",
    ])
    df = pd.read_csv(csv_path)

    # Handle both cleaned and raw column naming.
    date_col = "date" if "date" in df.columns else "Date"
    symbol_col = "symbol" if "symbol" in df.columns else "SYMBOL"
    wt_col = "weight_pct" if "weight_pct" in df.columns else "IDX WT %"
    price_col = "price" if "price" in df.columns else "PRICE"
    vol_col = "volume" if "volume" in df.columns else "VOLUME"

    df[date_col] = pd.to_datetime(df[date_col])
    df[wt_col] = pd.to_numeric(df[wt_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    df = df.sort_values([symbol_col, date_col]).dropna(subset=[symbol_col, date_col])

    reb_monthly = (
        df.groupby([pd.Grouper(key=date_col, freq="ME"), symbol_col])
        .last()
        .reset_index()
    )
    reb_monthly = reb_monthly[reb_monthly[date_col].dt.month.isin([3, 9])].copy()
    reb_monthly = reb_monthly.sort_values([date_col, symbol_col])

    reb_monthly["ret_21d"] = reb_monthly.groupby(symbol_col)[price_col].pct_change()
    reb_monthly["wt_rank"] = reb_monthly.groupby(date_col)[wt_col].rank(ascending=False, method="dense")

    daily = df[[date_col, symbol_col, price_col, vol_col]].copy()
    daily["ret_d"] = daily.groupby(symbol_col)[price_col].pct_change()
    daily["vol_63d"] = (
        daily.groupby(symbol_col)["ret_d"]
        .rolling(63, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    daily["turn_63d"] = (
        daily.groupby(symbol_col)[vol_col]
        .rolling(63, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    daily_last = daily.groupby([pd.Grouper(key=date_col, freq="ME"), symbol_col]).last().reset_index()
    daily_last = daily_last[daily_last[date_col].dt.month.isin([3, 9])][[date_col, symbol_col, "vol_63d", "turn_63d"]]

    panel = reb_monthly.merge(daily_last, on=[date_col, symbol_col], how="left")
    panel = panel.rename(columns={date_col: "Date", symbol_col: "SYMBOL", wt_col: "IDX WT %"})
    return panel


def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(panel["Date"].dropna().unique())
    rows = []

    for i in range(len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        cur = panel[panel["Date"] == d].copy()
        nxt = panel[panel["Date"] == d_next][["SYMBOL", "IDX WT %"]].copy()
        nxt = nxt.rename(columns={"IDX WT %": "next_weight"})
        cur = cur.merge(nxt, on="SYMBOL", how="left")
        cur["next_weight"] = cur["next_weight"].fillna(0.0)
        cur["stay_next"] = (cur["next_weight"] > 0).astype(int)
        cur["weight_change_next"] = cur["next_weight"] - cur["IDX WT %"]
        cur["next_rebalance_date"] = d_next
        rows.append(cur)

    return pd.concat(rows, ignore_index=True)


def attach_fund_features(ds: pd.DataFrame, fund_feats: pd.DataFrame) -> pd.DataFrame:
    ff = fund_feats.reset_index()
    if "Date" not in ff.columns:
        ff = ff.rename(columns={ff.columns[0]: "Date"})
    return ds.merge(ff, on="Date", how="left")


def attach_inflation_features(ds: pd.DataFrame, inflation_feats: pd.DataFrame) -> pd.DataFrame:
    out = ds.copy()
    out["year"] = out["Date".strip()].dt.year
    return out.merge(inflation_feats, on="year", how="left")


def fit_and_score_target_cycle(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep_cols = ["Date", "next_rebalance_date", "SYMBOL", "IDX WT %", "stay_next", "weight_change_next"]
    drop_cols = [
        "ISIN", "COMPANY", "FF BASED SHARES", "FF BASED MCAP", "ORD SHARES", "ORD SHARES MCAP",
        "PRICE", "VOLUME", "next_weight",
    ]
    feature_cols = [
        c for c in df.columns
        if c not in keep_cols + drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    data = df.dropna(subset=["stay_next", "weight_change_next"]).copy().sort_values("Date")

    train = data[data["next_rebalance_date"] <= CUTOFF_DATE].copy()
    target = data[data["next_rebalance_date"] == TARGET_NEXT_REBALANCE].copy()

    if train.empty:
        raise ValueError("No training rows after applying cutoff.")
    if target.empty:
        raise ValueError(f"No rows found for target rebalance date {TARGET_NEXT_REBALANCE.date()}.")
    if train["Date"].max() > CUTOFF_DATE:
        raise ValueError("Leakage detected: training features include dates after cutoff.")
    if target["Date"].max() > CUTOFF_DATE:
        raise ValueError("Leakage detected: target-cycle features include dates after cutoff.")

    train_medians = train[feature_cols].median()
    X_train = train[feature_cols].fillna(train_medians)
    X_target = target[feature_cols].fillna(train_medians)

    y_cls_train = train["stay_next"]
    y_reg_train = train["weight_change_next"]

    cls = RandomForestClassifier(n_estimators=500, random_state=42, min_samples_leaf=2, class_weight="balanced")
    reg = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)

    cls.fit(X_train, y_cls_train)
    reg.fit(X_train, y_reg_train)

    pred_cls = cls.predict(X_target)
    pred_prob = cls.predict_proba(X_target)[:, 1]
    pred_reg = reg.predict(X_target)

    out = target[["Date", "next_rebalance_date", "SYMBOL", "IDX WT %", "stay_next", "weight_change_next", "next_weight"]].copy()
    out["pred_stay_next"] = pred_cls
    out["pred_stay_prob"] = pred_prob
    out["pred_weight_change_next"] = pred_reg
    out["pred_next_weight"] = np.maximum(0.0, out["IDX WT %"] + out["pred_weight_change_next"])

    y_cls_true = out["stay_next"]
    y_reg_true = out["weight_change_next"]

    metrics = {
        "assumptions": {
            "rebalance_event_requested": "2026-03-15",
            "rebalance_proxy_in_data": str(TARGET_NEXT_REBALANCE.date()),
            "training_cutoff": str(CUTOFF_DATE.date()),
            "feature_snapshot_date_for_this_cycle": str(pd.Timestamp(out["Date"].iloc[0]).date()),
            "max_training_feature_date": str(pd.Timestamp(train["Date"].max()).date()),
            "max_target_feature_date": str(pd.Timestamp(target["Date"].max()).date()),
        },
        "sample_sizes": {
            "n_train": int(len(train)),
            "n_target_cycle": int(len(target)),
        },
        "classification": {
            "accuracy": float(accuracy_score(y_cls_true, pred_cls)),
            "f1": float(f1_score(y_cls_true, pred_cls, zero_division=0)),
            "actual_stay_rate": float(y_cls_true.mean()),
            "predicted_stay_rate": float(np.mean(pred_cls)),
        },
        "regression": {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_true, pred_reg))),
            "mae": float(mean_absolute_error(y_reg_true, pred_reg)),
            "r2": float(r2_score(y_reg_true, pred_reg)),
        },
        "top_features_classifier": [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(zip(feature_cols, cls.feature_importances_), key=lambda x: x[1], reverse=True)[:15]
        ],
        "top_features_regressor": [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: x[1], reverse=True)[:15]
        ],
    }

    return out.sort_values(["pred_stay_prob", "pred_next_weight"], ascending=[False, False]), metrics


def make_plots(march_eval: pd.DataFrame) -> None:
    plot_df = march_eval.copy()

    plt.figure(figsize=(10, 5))
    ordered = plot_df.sort_values("pred_stay_prob", ascending=False)
    x = np.arange(len(ordered))
    colors = ["#2e7d32" if v == 1 else "#c62828" for v in ordered["stay_next"]]
    plt.bar(x, ordered["pred_stay_prob"], color=colors)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, ordered["SYMBOL"], rotation=60, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Predicted Retention Probability (Mar 2026 Cycle)")
    plt.xlabel("Symbol")
    plt.ylabel("Predicted Stay Probability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "march_2026_retention_probability.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(plot_df["weight_change_next"], plot_df["pred_weight_change_next"], alpha=0.75)
    mn = min(plot_df["weight_change_next"].min(), plot_df["pred_weight_change_next"].min())
    mx = max(plot_df["weight_change_next"].max(), plot_df["pred_weight_change_next"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="black")
    plt.title("Actual vs Predicted Weight Change (Mar 2026)")
    plt.xlabel("Actual Weight Change")
    plt.ylabel("Predicted Weight Change")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "march_2026_weight_change_scatter.png", dpi=180)
    plt.close()

    compare = plot_df[["SYMBOL", "next_weight", "pred_next_weight"]].copy()
    compare["abs_error"] = (compare["pred_next_weight"] - compare["next_weight"]).abs()
    top_err = compare.sort_values("abs_error", ascending=False).head(15).sort_values("abs_error", ascending=True)

    plt.figure(figsize=(10, 6))
    y = np.arange(len(top_err))
    plt.barh(y - 0.2, top_err["next_weight"], height=0.38, label="Actual next weight")
    plt.barh(y + 0.2, top_err["pred_next_weight"], height=0.38, label="Predicted next weight")
    plt.yticks(y, top_err["SYMBOL"])
    plt.title("Largest Next-Weight Errors (Mar 2026)")
    plt.xlabel("Weight (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "march_2026_top_weight_errors.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()

    fund = load_fund_features()
    inflation = load_inflation_features()
    panel = build_rebalance_panel()
    ds = build_targets(panel)
    ds = attach_fund_features(ds, fund)
    ds = attach_inflation_features(ds, inflation)

    march_eval, metrics = fit_and_score_target_cycle(ds)
    make_plots(march_eval)

    ds.to_csv(OUT_DIR / "kse30_rebalance_training_panel_with_targets.csv", index=False)
    march_eval.to_csv(OUT_DIR / "march_2026_rebalance_actual_vs_pred.csv", index=False)
    (OUT_DIR / "march_2026_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("March 2026 backtest completed.")
    print(f"Evaluation table: {OUT_DIR / 'march_2026_rebalance_actual_vs_pred.csv'}")
    print(f"Metrics JSON: {OUT_DIR / 'march_2026_metrics.json'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
