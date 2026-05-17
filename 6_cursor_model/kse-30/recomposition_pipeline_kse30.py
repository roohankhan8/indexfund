"""
KSE-30 recomposition pipeline aligned to .ai-guidance/kse-30-index-methodology.md

Outputs are written to this folder:
  - recomposition_panel.csv
  - recomposition_constituents.csv
  - recomposition_events.csv

Run from repo root:
  python 6_cursor_model/kse-30/recomposition_pipeline_kse30.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "5_claude_pipeline" / "kse30_daily_data.csv"
OUT_DIR = Path(__file__).resolve().parent
CSV_DIR = OUT_DIR / "csvs"
FIG_DIR = OUT_DIR / "figs"

TOP_N = 30
MIN_LISTING_DAYS = 60
MIN_TRADING_RATIO = 0.75
MIN_FREE_FLOAT_PCT = 0.05

# Dataset does not contain direct PSX impact-cost/defaulters/CDC/suspension fields.
# We keep those flags explicit but non-blocking (documented in audit note).
REQUIRE_UNAVAILABLE_FLAGS = False


@dataclass(frozen=True)
class RecompCycle:
    as_of_date: pd.Timestamp
    trading_date: pd.Timestamp


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        out = pd.Series(np.ones(len(s)), index=s.index)
    else:
        out = (s - lo) / (hi - lo)
    return out if higher_is_better else (1.0 - out)


def nearest_trading_day(target: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    pos = np.argmin(np.abs((trading_days - target).days))
    return pd.Timestamp(trading_days[pos])


def build_cycles(trading_days: pd.DatetimeIndex) -> list[RecompCycle]:
    start_year, end_year = int(trading_days.min().year), int(trading_days.max().year)
    cycles: list[RecompCycle] = []
    for y in range(start_year, end_year + 1):
        for m, d in [(6, 30), (12, 31)]:
            as_of = pd.Timestamp(year=y, month=m, day=d)
            if as_of < trading_days.min() or as_of > trading_days.max():
                continue
            cycles.append(RecompCycle(as_of_date=as_of, trading_date=nearest_trading_day(as_of, trading_days)))
    cycles.sort(key=lambda x: x.as_of_date)
    return cycles


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df.rename(columns={
        "Date": "date",
        "SYMBOL": "symbol",
        "COMPANY": "company",
        "PRICE": "price",
        "IDX WT %": "weight_pct",
        "FF BASED SHARES": "free_float_shares",
        "FF BASED MCAP": "free_float_market_cap",
        "ORD SHARES": "ord_shares",
        "VOLUME": "volume",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "weight_pct", "free_float_shares", "free_float_market_cap", "ord_shares", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def compute_cycle_panel(df: pd.DataFrame, cycle: RecompCycle) -> pd.DataFrame:
    asof = cycle.trading_date
    hist_start = asof - pd.Timedelta(days=182)
    hist = df[(df["date"] >= hist_start) & (df["date"] <= asof)].copy()
    snap = df[df["date"] == asof].copy()

    total_days = hist["date"].nunique()

    rows = []
    for sym, s_hist in hist.groupby("symbol"):
        s_snap = snap[snap["symbol"] == sym]
        if s_snap.empty:
            continue

        first_date = s_hist["date"].min()
        trading_days = s_hist["date"].nunique()
        trading_ratio = trading_days / total_days if total_days else np.nan

        free_float_shares = float(s_snap["free_float_shares"].iloc[-1])
        ord_shares = float(s_snap["ord_shares"].iloc[-1]) if pd.notna(s_snap["ord_shares"].iloc[-1]) else np.nan
        ff_pct = (free_float_shares / ord_shares) if (pd.notna(ord_shares) and ord_shares > 0) else np.nan

        rows.append({
            "as_of_date": cycle.as_of_date,
            "trading_date": asof,
            "symbol": sym,
            "company": s_snap["company"].iloc[-1],
            "closing_price": float(s_snap["price"].iloc[-1]),
            "current_weight_pct": float(s_snap["weight_pct"].iloc[-1]),
            "free_float_shares": free_float_shares,
            "ord_shares": ord_shares,
            "free_float_pct": ff_pct,
            "free_float_market_cap": float(s_snap["free_float_market_cap"].iloc[-1]),
            "avg_volume_6m": float(s_hist["volume"].mean()),
            "trading_days_ratio": trading_ratio,
            "listing_days_in_sample": int((asof - first_date).days),
            "is_defaulter_unknown": np.nan,
            "is_suspended_unknown": np.nan,
            "is_cdc_eligible_unknown": np.nan,
            "impact_cost_avg_6m_unknown": np.nan,
        })

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    panel["elig_listing_age"] = panel["listing_days_in_sample"] >= MIN_LISTING_DAYS
    panel["elig_free_float_min"] = panel["free_float_pct"] >= MIN_FREE_FLOAT_PCT
    panel["elig_trading_ratio"] = panel["trading_days_ratio"] >= MIN_TRADING_RATIO

    if REQUIRE_UNAVAILABLE_FLAGS:
        panel["elig_defaulter"] = False
        panel["elig_suspension"] = False
        panel["elig_cdc"] = False
        panel["elig_impact_cost"] = False
    else:
        panel["elig_defaulter"] = True
        panel["elig_suspension"] = True
        panel["elig_cdc"] = True
        panel["elig_impact_cost"] = True

    panel["eligible"] = (
        panel["elig_listing_age"]
        & panel["elig_free_float_min"]
        & panel["elig_trading_ratio"]
        & panel["elig_defaulter"]
        & panel["elig_suspension"]
        & panel["elig_cdc"]
        & panel["elig_impact_cost"]
    )

    elig = panel[panel["eligible"]].copy()
    if not elig.empty:
        elig["score_ff_mcap"] = minmax(elig["free_float_market_cap"], higher_is_better=True)
        # Liquidity proxy only: higher avg volume => better liquidity => lower effective impact cost.
        elig["score_liquidity_proxy"] = minmax(elig["avg_volume_6m"], higher_is_better=True)
        elig["final_score"] = 0.5 * elig["score_ff_mcap"] + 0.5 * elig["score_liquidity_proxy"]
        elig = elig.sort_values(["final_score", "free_float_market_cap"], ascending=[False, False]).reset_index(drop=True)
        elig["rank"] = np.arange(1, len(elig) + 1)
        elig["is_constituent"] = elig["rank"] <= TOP_N

        panel = panel.merge(
            elig[["symbol", "score_ff_mcap", "score_liquidity_proxy", "final_score", "rank", "is_constituent"]],
            on="symbol",
            how="left",
        )
    else:
        panel["score_ff_mcap"] = np.nan
        panel["score_liquidity_proxy"] = np.nan
        panel["final_score"] = np.nan
        panel["rank"] = np.nan
        panel["is_constituent"] = False

    return panel


def metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae}


def build_transition_dataset(cons_all: pd.DataFrame, panel_all: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(panel_all["as_of_date"].unique()))
    rows = []
    for i, as_of in enumerate(dates[:-1]):
        next_as_of = dates[i + 1]
        cur_p = panel_all[pd.to_datetime(panel_all["as_of_date"]) == as_of].copy()
        nxt_p = panel_all[pd.to_datetime(panel_all["as_of_date"]) == next_as_of].copy()
        nxt_w = nxt_p.set_index("symbol")["current_weight_pct"].to_dict()
        nxt_const = set(
            nxt_p.loc[nxt_p["is_constituent"] == True, "symbol"].tolist()
        )
        cur_const = cur_p.loc[cur_p["is_constituent"] == True].copy()
        for _, r in cur_const.iterrows():
            sym = r["symbol"]
            rows.append(
                {
                    "as_of_date": as_of,
                    "next_as_of_date": next_as_of,
                    "symbol": sym,
                    "current_weight_pct": float(r["current_weight_pct"]),
                    "free_float_market_cap": float(r["free_float_market_cap"]),
                    "avg_volume_6m": float(r["avg_volume_6m"]),
                    "trading_days_ratio": float(r["trading_days_ratio"]),
                    "free_float_pct": float(r["free_float_pct"]) if pd.notna(r["free_float_pct"]) else np.nan,
                    "score_ff_mcap": float(r["score_ff_mcap"]) if pd.notna(r["score_ff_mcap"]) else np.nan,
                    "score_liquidity_proxy": float(r["score_liquidity_proxy"]) if pd.notna(r["score_liquidity_proxy"]) else np.nan,
                    "final_score": float(r["final_score"]) if pd.notna(r["final_score"]) else np.nan,
                    "rank": float(r["rank"]) if pd.notna(r["rank"]) else np.nan,
                    "retained": 1 if sym in nxt_const else 0,
                    "target_weight_pct": float(nxt_w.get(sym, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def run_proxy_prediction(panel_all: pd.DataFrame, cons_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = build_transition_dataset(cons_all, panel_all)
    if tr.empty:
        return pd.DataFrame(), pd.DataFrame()

    feat_cols = [
        "current_weight_pct",
        "free_float_market_cap",
        "avg_volume_6m",
        "trading_days_ratio",
        "free_float_pct",
        "score_ff_mcap",
        "score_liquidity_proxy",
        "final_score",
        "rank",
    ]
    tr[feat_cols] = tr[feat_cols].replace([np.inf, -np.inf], np.nan)

    cycles = sorted(pd.to_datetime(tr["as_of_date"].unique()))
    test_cycles = set(cycles[-2:]) if len(cycles) >= 3 else set(cycles[-1:])
    train = tr[~pd.to_datetime(tr["as_of_date"]).isin(test_cycles)].copy()
    test = tr[pd.to_datetime(tr["as_of_date"]).isin(test_cycles)].copy()
    if train.empty:
        train = tr.copy()
        test = tr.copy()

    med = train[feat_cols].median()
    train[feat_cols] = train[feat_cols].fillna(med)
    test[feat_cols] = test[feat_cols].fillna(med)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train[feat_cols].values)
    x_te = scaler.transform(test[feat_cols].values)

    y_ret_tr = train["retained"].values
    y_ret_te = test["retained"].values
    y_w_tr = train["target_weight_pct"].values
    y_w_te = test["target_weight_pct"].values

    logit = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(x_tr, y_ret_tr)
    rf_c = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=2, random_state=42).fit(x_tr, y_ret_tr)
    ridge = Ridge(alpha=1.0).fit(x_tr, y_w_tr)
    rf_r = RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=2, random_state=42).fit(x_tr, y_w_tr)

    p_log = logit.predict_proba(x_te)[:, 1]
    p_rf = rf_c.predict_proba(x_te)[:, 1]
    p_avg = (p_log + p_rf) / 2.0
    y_cls = (p_avg >= 0.5).astype(int)
    w_ridge = ridge.predict(x_te)
    w_rf = rf_r.predict(x_te)
    w_avg = (w_ridge + w_rf) / 2.0

    eval_rows = [
        {
            "task": "inclusion",
            "model": "avg(logit,rf)",
            "accuracy": round(float(accuracy_score(y_ret_te, y_cls)), 4),
            "auc": round(float(roc_auc_score(y_ret_te, p_avg)) if len(np.unique(y_ret_te)) > 1 else np.nan, 4),
        },
        {
            "task": "weight",
            "model": "avg(ridge,rf)",
            "rmse": round(metrics_reg(y_w_te, w_avg)["RMSE"], 4),
            "mae": round(metrics_reg(y_w_te, w_avg)["MAE"], 4),
        },
    ]
    eval_df = pd.DataFrame(eval_rows)

    latest = panel_all[pd.to_datetime(panel_all["as_of_date"]) == max(cycles)].copy()
    latest = latest[latest["is_constituent"] == True].copy()
    latest[feat_cols] = latest[feat_cols].fillna(med)
    x_f = scaler.transform(latest[feat_cols].values)

    latest["ret_prob_logit"] = logit.predict_proba(x_f)[:, 1]
    latest["ret_prob_rf"] = rf_c.predict_proba(x_f)[:, 1]
    latest["ret_prob_avg"] = (latest["ret_prob_logit"] + latest["ret_prob_rf"]) / 2.0
    latest["pred_weight_ridge"] = ridge.predict(x_f)
    latest["pred_weight_rf"] = rf_r.predict(x_f)
    latest["pred_weight_avg"] = (latest["pred_weight_ridge"] + latest["pred_weight_rf"]) / 2.0
    latest["pred_weight_change"] = latest["pred_weight_avg"] - latest["current_weight_pct"]
    latest["exclusion_risk"] = 1.0 - latest["ret_prob_avg"]
    latest = latest.sort_values("exclusion_risk", ascending=False)

    return eval_df, latest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    trading_days = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    cycles = build_cycles(trading_days)

    all_panels = []
    constituents_rows = []

    for cycle in cycles:
        panel = compute_cycle_panel(df, cycle)
        if panel.empty:
            continue
        all_panels.append(panel)

        const = panel[panel["is_constituent"] == True].copy()
        const = const.sort_values("rank")
        const["n_constituents"] = len(const)
        constituents_rows.append(const)

    panel_all = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()
    cons_all = pd.concat(constituents_rows, ignore_index=True) if constituents_rows else pd.DataFrame()

    events = []
    if not cons_all.empty:
        for as_of, grp in cons_all.groupby("as_of_date"):
            curr = set(grp["symbol"])
            prev_dates = sorted(d for d in cons_all["as_of_date"].unique() if d < as_of)
            prev = set(cons_all[cons_all["as_of_date"] == prev_dates[-1]]["symbol"]) if prev_dates else set()
            incoming = sorted(list(curr - prev))
            outgoing = sorted(list(prev - curr))
            events.append({
                "as_of_date": as_of,
                "incoming_count": len(incoming),
                "outgoing_count": len(outgoing),
                "incoming": ",".join(incoming),
                "outgoing": ",".join(outgoing),
            })

    events_df = pd.DataFrame(events)
    eval_df, forecast_df = run_proxy_prediction(panel_all, cons_all)

    panel_all.to_csv(CSV_DIR / "recomposition_panel.csv", index=False)
    cons_all.to_csv(CSV_DIR / "recomposition_constituents.csv", index=False)
    events_df.to_csv(CSV_DIR / "recomposition_events.csv", index=False)
    eval_df.to_csv(CSV_DIR / "results_inclusion_weight_eval.csv", index=False)
    forecast_df.to_csv(CSV_DIR / "results_inclusion_weight_forecast.csv", index=False)

    # Figure 1: incoming/outgoing counts by cycle
    if not events_df.empty:
        plot_df = events_df.copy()
        plot_df["as_of_date"] = pd.to_datetime(plot_df["as_of_date"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_df["as_of_date"], plot_df["incoming_count"], marker="o", label="Incoming")
        ax.plot(plot_df["as_of_date"], plot_df["outgoing_count"], marker="o", label="Outgoing")
        ax.set_title("KSE-30 Recomposition Counts by Cycle")
        ax.set_xlabel("As-of Date")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "R01_recomposition_counts.png", dpi=150)
        plt.close(fig)

    # Figure 2: top-10 final scores for latest cycle
    if not cons_all.empty:
        latest_date = cons_all["as_of_date"].max()
        latest = cons_all[cons_all["as_of_date"] == latest_date].sort_values("rank").head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(latest["symbol"], latest["final_score"], color="#2c3e50", alpha=0.9)
        ax.set_title(f"Top-10 Final Scores ({pd.Timestamp(latest_date).date()})")
        ax.set_xlabel("Symbol")
        ax.set_ylabel("Final Score")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "R02_top10_final_scores_latest_cycle.png", dpi=150)
        plt.close(fig)

    # Figure 3: eligibility pass rate by cycle
    if not panel_all.empty:
        elig_cycle = (
            panel_all.groupby("as_of_date", as_index=False)
            .agg(total=("symbol", "count"), eligible=("eligible", "sum"))
        )
        elig_cycle["pass_rate"] = elig_cycle["eligible"] / elig_cycle["total"]
        elig_cycle["as_of_date"] = pd.to_datetime(elig_cycle["as_of_date"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(elig_cycle["as_of_date"], elig_cycle["pass_rate"], marker="o", color="#16a085")
        ax.set_title("Eligibility Pass Rate by Cycle")
        ax.set_xlabel("As-of Date")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "R03_eligibility_pass_rate.png", dpi=150)
        plt.close(fig)

    # Figure 4: predicted retention probabilities (latest cycle constituents)
    if not forecast_df.empty:
        p = forecast_df.sort_values("ret_prob_avg")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#e74c3c" if v < 0.60 else "#f39c12" if v < 0.80 else "#2ecc71" for v in p["ret_prob_avg"]]
        ax.barh(p["symbol"], p["ret_prob_avg"], color=colors, alpha=0.9)
        ax.set_title("Proxy Inclusion Prediction: Retention Probability")
        ax.set_xlabel("Retention Probability")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "R04_retention_probability_proxy.png", dpi=150)
        plt.close(fig)

    # Figure 5: predicted weight changes
    if not forecast_df.empty:
        w = forecast_df.sort_values("pred_weight_change")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#c0392b" if v < 0 else "#1e8449" for v in w["pred_weight_change"]]
        ax.barh(w["symbol"], w["pred_weight_change"], color=colors, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Proxy Weight Prediction: Expected Weight Change")
        ax.set_xlabel("Predicted Weight Change (%)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "R05_weight_change_proxy.png", dpi=150)
        plt.close(fig)

    print(f"Saved: {CSV_DIR / 'recomposition_panel.csv'}")
    print(f"Saved: {CSV_DIR / 'recomposition_constituents.csv'}")
    print(f"Saved: {CSV_DIR / 'recomposition_events.csv'}")
    print(f"Saved: {CSV_DIR / 'results_inclusion_weight_eval.csv'}")
    print(f"Saved: {CSV_DIR / 'results_inclusion_weight_forecast.csv'}")
    print(f"Saved: {FIG_DIR / 'R01_recomposition_counts.png'}")
    print(f"Saved: {FIG_DIR / 'R02_top10_final_scores_latest_cycle.png'}")
    print(f"Saved: {FIG_DIR / 'R03_eligibility_pass_rate.png'}")
    print(f"Saved: {FIG_DIR / 'R04_retention_probability_proxy.png'}")
    print(f"Saved: {FIG_DIR / 'R05_weight_change_proxy.png'}")


if __name__ == "__main__":
    main()
