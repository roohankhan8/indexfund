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
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
TABLE_DIR = OUT_DIR / "tables"
METRIC_DIR = OUT_DIR / "metrics"
FIG_DIR = OUT_DIR / "figures"
FUNDS = ["AKD", "NBP", "NTI"]


def ensure_dirs() -> None:
    for p in [OUT_DIR, TABLE_DIR, METRIC_DIR, FIG_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def compute_flow(df: pd.DataFrame) -> pd.Series:
    nav_lag = df["NAV"].shift(1)
    aum_lag = df["AUM"].shift(1)
    return df["AUM"] - aum_lag * (df["NAV"] / nav_lag)


def load_fund_features() -> pd.DataFrame:
    xls = pd.ExcelFile(DATA_DIR / "funds_data.xlsx")
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


def build_rebalance_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "kse30_daily_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["IDX WT %"] = pd.to_numeric(df["IDX WT %"], errors="coerce")
    df = df.sort_values(["SYMBOL", "Date"]).dropna(subset=["SYMBOL", "Date"])

    # KSE-30 rebalancing treated as semiannual (Mar/Sep).
    reb_monthly = (
        df.groupby([pd.Grouper(key="Date", freq="ME"), "SYMBOL"])  # last available row in month per symbol
        .last()
        .reset_index()
    )
    reb_monthly = reb_monthly[reb_monthly["Date"].dt.month.isin([3, 9])].copy()
    reb_monthly = reb_monthly.sort_values(["Date", "SYMBOL"])

    reb_monthly["ret_21d"] = reb_monthly.groupby("SYMBOL")["PRICE"].pct_change()
    reb_monthly["wt_rank"] = reb_monthly.groupby("Date")["IDX WT %"].rank(ascending=False, method="dense")

    # volatility/turnover proxy from preceding 63 business days for each symbol
    daily = df[["Date", "SYMBOL", "PRICE", "VOLUME"]].copy()
    daily["ret_d"] = daily.groupby("SYMBOL")["PRICE"].pct_change()
    daily["vol_63d"] = (
        daily.groupby("SYMBOL")["ret_d"]
        .rolling(63, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    daily["turn_63d"] = (
        daily.groupby("SYMBOL")["VOLUME"]
        .rolling(63, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    daily_last = daily.groupby([pd.Grouper(key="Date", freq="ME"), "SYMBOL"]).last().reset_index()
    daily_last = daily_last[daily_last["Date"].dt.month.isin([3, 9])][["Date", "SYMBOL", "vol_63d", "turn_63d"]]

    panel = reb_monthly.merge(daily_last, on=["Date", "SYMBOL"], how="left")
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
    ff = fund_feats.copy()
    ff = ff.reset_index()
    if "Date" not in ff.columns:
        ff = ff.rename(columns={ff.columns[0]: "Date"})
    out = ds.merge(ff, on="Date", how="left")
    return out


def train_and_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str], np.ndarray, np.ndarray]:
    keep_cols = [
        "Date", "next_rebalance_date", "SYMBOL", "IDX WT %", "stay_next", "weight_change_next",
    ]
    feature_cols = [
        c for c in df.columns
        if c not in keep_cols + ["ISIN", "COMPANY", "FF BASED SHARES", "FF BASED MCAP", "ORD SHARES", "ORD SHARES MCAP", "PRICE", "VOLUME", "next_weight"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    data = df.dropna(subset=["stay_next", "weight_change_next"]).copy()
    data = data.sort_values("Date")

    unique_dates = sorted(data["Date"].unique())
    test_dates = unique_dates[-2:]
    train = data[~data["Date"].isin(test_dates)].copy()
    test = data[data["Date"].isin(test_dates)].copy()

    train_medians = train[feature_cols].median()
    X_train = train[feature_cols].fillna(train_medians)
    X_test = test[feature_cols].fillna(train_medians)
    y_cls_train, y_cls_test = train["stay_next"], test["stay_next"]
    y_reg_train, y_reg_test = train["weight_change_next"], test["weight_change_next"]

    cls = RandomForestClassifier(n_estimators=500, random_state=42, min_samples_leaf=2, class_weight="balanced")
    cls.fit(X_train, y_cls_train)
    pred_cls = cls.predict(X_test)
    pred_prob = cls.predict_proba(X_test)[:, 1]

    reg = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)
    reg.fit(X_train, y_reg_train)
    pred_reg = reg.predict(X_test)

    out = test[["Date", "next_rebalance_date", "SYMBOL", "IDX WT %", "stay_next", "weight_change_next"]].copy()
    out["pred_stay_next"] = pred_cls
    out["pred_stay_prob"] = pred_prob
    out["pred_weight_change_next"] = pred_reg
    out["pred_next_weight"] = np.maximum(0.0, out["IDX WT %"] + out["pred_weight_change_next"])

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_dates": [str(pd.Timestamp(d).date()) for d in test_dates],
        "classification": {
            "accuracy": float(accuracy_score(y_cls_test, pred_cls)),
            "f1": float(f1_score(y_cls_test, pred_cls, zero_division=0)),
            "positive_rate_actual": float(y_cls_test.mean()),
            "positive_rate_pred": float(np.mean(pred_cls)),
        },
        "regression": {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, pred_reg))),
            "mae": float(mean_absolute_error(y_reg_test, pred_reg)),
            "r2": float(r2_score(y_reg_test, pred_reg)),
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

    return out, metrics, feature_cols, cls.feature_importances_, reg.feature_importances_


def score_latest_unlabeled(df_full: pd.DataFrame, labeled_cols: list[str]) -> pd.DataFrame:
    feature_cols = [
        c for c in df_full.columns
        if c not in labeled_cols + ["ISIN", "COMPANY", "FF BASED SHARES", "FF BASED MCAP", "ORD SHARES", "ORD SHARES MCAP", "PRICE", "VOLUME", "next_weight"]
        and pd.api.types.is_numeric_dtype(df_full[c])
    ]

    labeled = df_full.dropna(subset=["stay_next", "weight_change_next"]).copy().sort_values("Date")
    unlabeled = df_full[df_full["stay_next"].isna()].copy()
    train_medians = labeled[feature_cols].median()

    cls = RandomForestClassifier(n_estimators=500, random_state=42, min_samples_leaf=2, class_weight="balanced")
    reg = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)
    cls.fit(labeled[feature_cols].fillna(train_medians), labeled["stay_next"])
    reg.fit(labeled[feature_cols].fillna(train_medians), labeled["weight_change_next"])

    unlabeled["pred_stay_next"] = cls.predict(unlabeled[feature_cols].fillna(train_medians))
    unlabeled["pred_stay_prob"] = cls.predict_proba(unlabeled[feature_cols].fillna(train_medians))[:, 1]
    unlabeled["pred_weight_change_next"] = reg.predict(unlabeled[feature_cols].fillna(train_medians))
    unlabeled["pred_next_weight"] = np.maximum(0.0, unlabeled["IDX WT %"] + unlabeled["pred_weight_change_next"])

    cols = ["Date", "SYMBOL", "IDX WT %", "pred_stay_next", "pred_stay_prob", "pred_weight_change_next", "pred_next_weight"]
    return unlabeled[cols].sort_values(["pred_stay_prob", "pred_next_weight"], ascending=[False, False])


def make_plot(pred: pd.DataFrame) -> None:
    plot_df = pred.groupby("Date")[["stay_next", "pred_stay_next"]].mean().reset_index()
    plt.figure(figsize=(9, 4.5))
    plt.plot(plot_df["Date"], plot_df["stay_next"], label="Actual stay rate")
    plt.plot(plot_df["Date"], plot_df["pred_stay_next"], label="Predicted stay rate", linestyle="--")
    plt.title("KSE-30 Rebalance: Stay Rate by Rebalance Date")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Share of Constituents Staying")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_stay_rate_actual_vs_pred.png", dpi=180)
    plt.close()


def make_all_plots(
    pred_test: pd.DataFrame,
    forecast_next: pd.DataFrame,
    feature_cols: list[str],
    cls_importance: np.ndarray,
    reg_importance: np.ndarray,
) -> None:
    make_plot(pred_test)

    cm = confusion_matrix(pred_test["stay_next"], pred_test["pred_stay_next"], labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Stay vs Excluded)")
    plt.xticks([0, 1], ["Pred Excl", "Pred Stay"])
    plt.yticks([0, 1], ["Actual Excl", "Actual Stay"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(pred_test["weight_change_next"], pred_test["pred_weight_change_next"], alpha=0.7)
    lim = [
        min(pred_test["weight_change_next"].min(), pred_test["pred_weight_change_next"].min()),
        max(pred_test["weight_change_next"].max(), pred_test["pred_weight_change_next"].max()),
    ]
    plt.plot(lim, lim, linestyle="--")
    plt.title("Actual vs Predicted Weight Change")
    plt.xlabel("Actual Weight Change")
    plt.ylabel("Predicted Weight Change")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_weight_change_actual_vs_pred.png", dpi=180)
    plt.close()

    by_date = pred_test.groupby("Date")[["weight_change_next", "pred_weight_change_next"]].mean().reset_index()
    plt.figure(figsize=(8, 4.5))
    plt.plot(by_date["Date"], by_date["weight_change_next"], label="Actual avg change")
    plt.plot(by_date["Date"], by_date["pred_weight_change_next"], linestyle="--", label="Pred avg change")
    plt.title("Average Weight Change by Rebalance Date")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Average Weight Change")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_avg_weight_change_by_date.png", dpi=180)
    plt.close()

    cls_top = sorted(zip(feature_cols, cls_importance), key=lambda x: x[1], reverse=True)[:10]
    reg_top = sorted(zip(feature_cols, reg_importance), key=lambda x: x[1], reverse=True)[:10]

    plt.figure(figsize=(8, 5))
    names = [x[0] for x in cls_top][::-1]
    vals = [x[1] for x in cls_top][::-1]
    plt.barh(names, vals)
    plt.title("Top 10 Features: Stay/Exclude Classifier")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_top_features_classifier.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    names = [x[0] for x in reg_top][::-1]
    vals = [x[1] for x in reg_top][::-1]
    plt.barh(names, vals)
    plt.title("Top 10 Features: Weight-Change Regressor")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_top_features_regressor.png", dpi=180)
    plt.close()

    low_stay = forecast_next.sort_values("pred_stay_prob").head(12).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(low_stay["SYMBOL"], low_stay["pred_stay_prob"])
    plt.title("Lowest Predicted Stay Probability (Next Rebalance)")
    plt.xlabel("Symbol")
    plt.ylabel("Predicted Stay Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kse30_lowest_stay_prob_next_rebalance.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()

    fund = load_fund_features()
    panel = build_rebalance_panel()
    labeled = build_targets(panel)
    labeled = attach_fund_features(labeled, fund)

    pred_test, metrics, feature_cols, cls_importance, reg_importance = train_and_evaluate(labeled)

    # Attach empty targets for latest date so we can score upcoming rebalance from current constituents.
    latest = panel[panel["Date"] == panel["Date"].max()].copy()
    latest["next_rebalance_date"] = pd.NaT
    latest["next_weight"] = np.nan
    latest["stay_next"] = np.nan
    latest["weight_change_next"] = np.nan
    full = pd.concat([labeled, attach_fund_features(latest, fund)], ignore_index=True, sort=False)

    forecast_next = score_latest_unlabeled(
        full,
        labeled_cols=["Date", "next_rebalance_date", "SYMBOL", "IDX WT %", "stay_next", "weight_change_next"],
    )
    make_all_plots(pred_test, forecast_next, feature_cols, cls_importance, reg_importance)

    labeled.to_csv(TABLE_DIR / "kse30_rebalance_training_panel.csv", index=False)
    pred_test.to_csv(TABLE_DIR / "kse30_rebalance_test_predictions.csv", index=False)
    forecast_next.to_csv(TABLE_DIR / "kse30_next_rebalance_forecast.csv", index=False)
    (METRIC_DIR / "kse30_rebalance_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("KSE-30 rebalance pipeline completed.")
    print(f"Train/test predictions: {TABLE_DIR / 'kse30_rebalance_test_predictions.csv'}")
    print(f"Next rebalance forecast: {TABLE_DIR / 'kse30_next_rebalance_forecast.csv'}")


if __name__ == "__main__":
    main()
