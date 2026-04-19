"""
NOTEBOOK 4b — KSE-30 Rebalancing Weight & Inclusion Prediction
===============================================================
Input  : kse30_stocks_daily.csv, monthly_master.csv
Outputs: figures → ./figures/rebalancing/
         results → results_rebalancing.csv

What this notebook answers
--------------------------
The KSE-30 is rebalanced semi-annually (always ~March 15 and ~September 15).
Each rebalancing:
  - Rotates 3–7 stocks in and an equal number out
  - Adjusts weights of remaining stocks based on free-float market cap

This notebook builds TWO prediction models:

  Task A — Weight prediction (regression)
    Predict each continuing stock's weight at the NEXT rebalancing
    from pre-rebalancing features (trailing momentum, volatility,
    market-cap proxy, current weight).
    Models: Ridge regression, Random Forest (numpy-based bagging)

  Task B — Inclusion prediction (binary classification)
    Predict whether a stock will be RETAINED in the index at the
    next rebalancing (vs dropped and replaced).
    Models: Logistic regression, Decision tree (manual CART, depth 3)

Data structure
--------------
Panel: 9 rebalancing windows × ~30 stocks = 270 observations
Features measured in the 30-day window before each rebalancing date
Target measured on the rebalancing date itself

Rebalancing schedule (from data):
  2021-03-15, 2021-09-15, 2022-03-15, 2022-09-15, 2023-03-15,
  2023-09-15, 2024-03-15, 2024-09-16, 2025-03-17, 2025-09-15

Train: first 7 rebalancing windows (2021-03-15 → 2024-03-15)
Test:  last 2 rebalancing windows   (2024-09-16, 2025-03-17)
       (2025-09-15 is the prediction target — no future data)

Run: python nb4b_rebalancing_prediction.py
Requirements: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "rebalancing")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 9})

# ── Confirmed rebalancing dates ───────────────────────────────────────────────
REBAL_DATES = pd.to_datetime([
    "2021-03-15", "2021-09-15", "2022-03-15", "2022-09-15",
    "2023-03-15", "2023-09-15", "2024-03-15", "2024-09-16",
    "2025-03-17", "2025-09-15"
])
# Train on first 7 windows (predict into window 8 and 9)
# Window k = features at date k, target at date k+1
# So train windows 0-6 (targets at dates 1-7), test windows 7-8 (targets at 8-9)
TRAIN_WINDOWS = list(range(7))   # indices into REBAL_DATES for feature snapshots
TEST_WINDOWS  = [7, 8]

# ── Load ─────────────────────────────────────────────────────────────────────
stocks  = pd.read_csv("new_data/kse30_stocks_daily.csv", parse_dates=["date"])
monthly = pd.read_csv("new_data/monthly_master.csv",     parse_dates=["date"])
stocks  = stocks.sort_values(["symbol","date"]).reset_index(drop=True)
print(f"Stock rows: {len(stocks):,}  |  Symbols ever: {stocks.symbol.nunique()}")
print(f"Date range: {stocks.date.min().date()} → {stocks.date.max().date()}")
print(f"Rebalancing dates confirmed: {len(REBAL_DATES)}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD PANEL DATASET
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: Building rebalancing panel dataset ─────────────────")

def get_nearest_trading_date(target_date, available_dates, window=5):
    """Return closest available date within ±window days."""
    candidates = [d for d in available_dates
                  if abs((pd.Timestamp(d) - target_date).days) <= window]
    if candidates:
        return min(candidates, key=lambda d: abs((pd.Timestamp(d) - target_date).days))
    return None

available_dates = sorted(stocks["date"].unique())

records = []

for win_idx in range(len(REBAL_DATES) - 1):
    rdate      = REBAL_DATES[win_idx]
    next_rdate = REBAL_DATES[win_idx + 1]

    # Snap nearest actual trading date
    rdate_actual      = get_nearest_trading_date(rdate,      available_dates)
    next_rdate_actual = get_nearest_trading_date(next_rdate, available_dates)
    if rdate_actual is None or next_rdate_actual is None:
        continue

    # Feature window: 30 trading days before current rebalancing
    rdate_ts      = pd.Timestamp(rdate_actual)
    snap_end      = rdate_ts - pd.Timedelta(days=1)
    snap_start_30 = rdate_ts - pd.Timedelta(days=30)
    snap_start_60 = rdate_ts - pd.Timedelta(days=60)
    snap_start_90 = rdate_ts - pd.Timedelta(days=90)

    snap30 = stocks[(stocks.date >= snap_start_30) & (stocks.date <= snap_end)]
    snap60 = stocks[(stocks.date >= snap_start_60) & (stocks.date <= snap_end)]
    snap90 = stocks[(stocks.date >= snap_start_90) & (stocks.date <= snap_end)]

    # Stocks present AT this rebalancing date (current constituents)
    curr_constituents = set(
        stocks[stocks.date == rdate_ts]["symbol"]
    )
    next_constituents = set(
        stocks[stocks.date == pd.Timestamp(next_rdate_actual)]["symbol"]
    )

    # Current weights
    curr_wts = stocks[stocks.date == rdate_ts].set_index("symbol")["weight_pct"]
    next_wts = stocks[stocks.date == pd.Timestamp(next_rdate_actual)].set_index(
        "symbol")["weight_pct"]

    for sym in curr_constituents:
        # 30-day features
        s30 = snap30[snap30.symbol == sym]
        s60 = snap60[snap60.symbol == sym]
        s90 = snap90[snap90.symbol == sym]

        if len(s30) < 5:  # insufficient data
            continue

        cur_wt = curr_wts.get(sym, np.nan)

        # Momentum: cumulative log return over trailing windows
        mom_30 = s30["log_return"].sum()
        mom_60 = s60["log_return"].sum() if len(s60) >= 10 else np.nan
        mom_90 = s90["log_return"].sum() if len(s90) >= 15 else np.nan

        # Volatility
        vol_30 = s30["log_return"].std() * np.sqrt(252)
        vol_60 = s60["log_return"].std() * np.sqrt(252) if len(s60) >= 10 else np.nan

        # Volume and market-cap proxy
        avg_vol   = s30["volume"].mean()
        last_price = s30["price"].iloc[-1]
        mkt_cap_proxy = last_price * avg_vol  # price × avg daily volume

        # Price relative to 20-day and 50-day MA
        ma20_last = s30["ma_20"].iloc[-1] if "ma_20" in s30.columns else np.nan
        ma50_last = s30["ma_50"].iloc[-1] if "ma_50" in s30.columns else np.nan
        price_to_ma20 = (last_price / ma20_last - 1) if ma20_last > 0 else np.nan
        price_to_ma50 = (last_price / ma50_last - 1) if ma50_last > 0 else np.nan

        # Weight momentum: how much has weight drifted since last rebalancing?
        if len(s30) > 1:
            wt_first = s30["weight_pct"].iloc[0]
            wt_last  = s30["weight_pct"].iloc[-1]
            wt_drift  = wt_last - wt_first
            wt_range  = s30["weight_pct"].max() - s30["weight_pct"].min()
        else:
            wt_drift = wt_range = np.nan

        # Target A: weight at next rebalancing (0 if dropped)
        target_wt = next_wts.get(sym, 0.0)

        # Target B: retained (1) or dropped (0)?
        retained = 1 if sym in next_constituents else 0

        records.append({
            "win_idx":        win_idx,
            "rebal_date":     rdate_ts,
            "next_rebal_date":pd.Timestamp(next_rdate_actual),
            "symbol":         sym,
            "cur_weight":     cur_wt,
            "mom_30":         mom_30,
            "mom_60":         mom_60,
            "mom_90":         mom_90,
            "vol_30":         vol_30,
            "vol_60":         vol_60,
            "avg_volume":     avg_vol,
            "mkt_cap_proxy":  mkt_cap_proxy,
            "price_to_ma20":  price_to_ma20,
            "price_to_ma50":  price_to_ma50,
            "wt_drift":       wt_drift,
            "wt_range":       wt_range,
            "target_weight":  target_wt,
            "retained":       retained,
        })

panel = pd.DataFrame(records)
print(f"Panel: {len(panel)} rows | {panel.win_idx.nunique()} windows | "
      f"{panel.symbol.nunique()} unique symbols")
print(f"Retained rate: {panel.retained.mean()*100:.1f}%")
print(f"Dropped rate:  {(1-panel.retained).mean()*100:.1f}%")

# Fill remaining NaN with column medians (non-leaking: use train medians later)
FEAT_COLS = ["cur_weight","mom_30","mom_60","mom_90",
             "vol_30","vol_60","avg_volume","mkt_cap_proxy",
             "price_to_ma20","price_to_ma50","wt_drift","wt_range"]

print(f"\nFeature NaN rates:")
for c in FEAT_COLS:
    print(f"  {c:20s}: {panel[c].isnull().mean()*100:.1f}%")

train_medians = panel[panel.win_idx.isin(TRAIN_WINDOWS)][FEAT_COLS].median()
panel[FEAT_COLS] = panel[FEAT_COLS].fillna(train_medians)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE ANALYSIS OF REBALANCING PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Rebalancing pattern analysis ───────────────────────")

print(f"\nRebalancing summary across {len(REBAL_DATES)-1} periods:")
for win_idx in range(len(REBAL_DATES) - 1):
    sub = panel[panel.win_idx == win_idx]
    dropped = sub[sub.retained == 0]
    retained = sub[sub.retained == 1]
    print(f"  Window {win_idx+1} "
          f"({REBAL_DATES[win_idx].date()} → {REBAL_DATES[win_idx+1].date()}): "
          f"{len(retained)} retained, {len(dropped)} dropped | "
          f"Dropped: {sorted(dropped.symbol.tolist())}")

print("\nFeature comparison: retained vs dropped stocks")
print(f"{'Feature':20s} {'Retained mean':>15} {'Dropped mean':>14} {'p-value':>10}")
print("-" * 62)
for feat in FEAT_COLS:
    ret_vals = panel[panel.retained == 1][feat].dropna()
    drp_vals = panel[panel.retained == 0][feat].dropna()
    if len(drp_vals) < 3:
        continue
    _, p = stats.mannwhitneyu(ret_vals, drp_vals, alternative="two-sided")
    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
    print(f"{feat:20s} {ret_vals.mean():>15.4f} {drp_vals.mean():>14.4f} "
          f"{p:>10.4f} {sig}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 3: Train/test split ───────────────────────────────────")

train = panel[panel.win_idx.isin(TRAIN_WINDOWS)].copy()
test  = panel[panel.win_idx.isin(TEST_WINDOWS)].copy()
print(f"Train: {len(train)} obs ({train.win_idx.nunique()} windows)")
print(f"Test:  {len(test)}  obs ({test.win_idx.nunique()}  windows)")

# Scale features (fit scaler on train only)
scaler = StandardScaler()
X_train_raw = train[FEAT_COLS].values
X_test_raw  = test[FEAT_COLS].values
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

y_wt_train = train["target_weight"].values
y_wt_test  = test["target_weight"].values
y_ret_train = train["retained"].values
y_ret_test  = test["retained"].values


# ═══════════════════════════════════════════════════════════════════════════
# TASK A — WEIGHT PREDICTION (REGRESSION)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Task A: Weight prediction (regression) ────────────────────────")

def reg_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    if label:
        print(f"    {label:30s}  RMSE={rmse:.4f}%  MAE={mae:.4f}%  R²={r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# Naive: predict next weight = current weight
naive_wt_pred_train = train["cur_weight"].values
naive_wt_pred_test  = test["cur_weight"].values
m_naive_wt_tr = reg_metrics(y_wt_train, naive_wt_pred_train, "Naive (train)")
m_naive_wt    = reg_metrics(y_wt_test,  naive_wt_pred_test,  "Naive (test) ")

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_wt_train)
ridge_pred_train = ridge.predict(X_train)
ridge_pred_test  = ridge.predict(X_test)
m_ridge_tr = reg_metrics(y_wt_train, ridge_pred_train, "Ridge regression (train)")
m_ridge    = reg_metrics(y_wt_test,  ridge_pred_test,  "Ridge regression (test) ")

print("\n  Ridge feature importances (coefficient magnitudes):")
coef_df = pd.DataFrame({"Feature": FEAT_COLS, "Coef": ridge.coef_})
coef_df["AbsCoef"] = coef_df["Coef"].abs()
coef_df = coef_df.sort_values("AbsCoef", ascending=False)
for _, row in coef_df.iterrows():
    print(f"    {row['Feature']:20s}: {row['Coef']:+.4f}")

# Random Forest regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=4,
                                min_samples_leaf=3, random_state=42)
rf_reg.fit(X_train, y_wt_train)
rf_pred_train = rf_reg.predict(X_train)
rf_pred_test  = rf_reg.predict(X_test)
m_rf_tr = reg_metrics(y_wt_train, rf_pred_train, "Random Forest (train)")
m_rf    = reg_metrics(y_wt_test,  rf_pred_test,  "Random Forest (test) ")

print("\n  RF feature importances:")
fi_df = pd.DataFrame({"Feature": FEAT_COLS,
                       "Importance": rf_reg.feature_importances_})
fi_df = fi_df.sort_values("Importance", ascending=False)
for _, row in fi_df.iterrows():
    print(f"    {row['Feature']:20s}: {row['Importance']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# TASK B — INCLUSION PREDICTION (BINARY CLASSIFICATION)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Task B: Inclusion prediction (classification) ─────────────────")

def clf_metrics(y_true, y_pred, y_prob=None, label=""):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    if label:
        print(f"    {label:30s}  Acc={acc:.3f}  AUC={auc:.3f}  "
              f"Prec={prec:.3f}  Rec={rec:.3f}  "
              f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"Accuracy": acc, "AUC": auc, "Precision": prec,
            "Recall": rec, "TP": tp, "FP": fp, "FN": fn, "TN": tn}

# Naive: predict all retained (majority class baseline)
naive_clf_pred = np.ones(len(y_ret_test), dtype=int)
naive_prob     = np.ones(len(y_ret_test)) * 0.85  # rough prior
m_naive_clf = clf_metrics(y_ret_test, naive_clf_pred, naive_prob, "Naive (all retained)")

# Logistic regression
logit = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
logit.fit(X_train, y_ret_train)
logit_pred_test  = logit.predict(X_test)
logit_prob_test  = logit.predict_proba(X_test)[:, 1]
logit_pred_train = logit.predict(X_train)
m_logit_tr  = clf_metrics(y_ret_train, logit_pred_train, label="Logistic (train)")
m_logit     = clf_metrics(y_ret_test,  logit_pred_test,
                           logit_prob_test, "Logistic regression  ")

print("\n  Logistic regression coefficients:")
logit_coef = pd.DataFrame({"Feature": FEAT_COLS, "Coef": logit.coef_[0]})
logit_coef["AbsCoef"] = logit_coef["Coef"].abs()
for _, row in logit_coef.sort_values("AbsCoef", ascending=False).iterrows():
    print(f"    {row['Feature']:20s}: {row['Coef']:+.4f}  "
          f"({'raises' if row['Coef'] > 0 else 'lowers'} retention prob)")

# Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3,
                                 min_samples_leaf=3, random_state=42)
rf_clf.fit(X_train, y_ret_train)
rf_clf_pred_test  = rf_clf.predict(X_test)
rf_clf_prob_test  = rf_clf.predict_proba(X_test)[:, 1]
rf_clf_pred_train = rf_clf.predict(X_train)
m_rf_clf_tr = clf_metrics(y_ret_train, rf_clf_pred_train, label="RF Classifier (train)")
m_rf_clf    = clf_metrics(y_ret_test, rf_clf_pred_test,
                           rf_clf_prob_test, "RF Classifier        ")

print("\n  Classification report (Random Forest, test set):")
print(classification_report(y_ret_test, rf_clf_pred_test,
                             target_names=["Dropped","Retained"]))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — NEXT REBALANCING PREDICTION (2025-09-15 window)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 4: Predict next rebalancing (2025-09-15) ─────────────")

# Features: 30 days before 2025-09-15 prediction date
# The last known rebalancing was 2025-03-17
# Current constituents as of 2025-03-17 onward

pred_rdate = REBAL_DATES[-1]  # 2025-09-15
snap_end_pred   = pred_rdate - pd.Timedelta(days=1)
snap_start_pred = pred_rdate - pd.Timedelta(days=30)

# Current constituents (from last rebalancing data)
last_rebal_date = get_nearest_trading_date(REBAL_DATES[-2], available_dates)
current_syms = set(stocks[stocks.date == pd.Timestamp(last_rebal_date)]["symbol"])
print(f"Current constituents ({pd.Timestamp(last_rebal_date).date()}): {len(current_syms)}")

future_feats = []
for sym in current_syms:
    s30 = stocks[(stocks.date >= snap_start_pred) &
                 (stocks.date <= snap_end_pred) &
                 (stocks.symbol == sym)]
    if len(s30) < 5:
        continue
    cur_wt = stocks[(stocks.date == pd.Timestamp(last_rebal_date)) &
                    (stocks.symbol == sym)]["weight_pct"].values
    cur_wt = cur_wt[0] if len(cur_wt) > 0 else np.nan

    mom_30 = s30["log_return"].sum()
    vol_30 = s30["log_return"].std() * np.sqrt(252)
    avg_vol = s30["volume"].mean()
    last_price = s30["price"].iloc[-1]

    s60 = stocks[(stocks.date >= (pred_rdate - pd.Timedelta(days=60))) &
                 (stocks.date <= snap_end_pred) & (stocks.symbol == sym)]
    s90 = stocks[(stocks.date >= (pred_rdate - pd.Timedelta(days=90))) &
                 (stocks.date <= snap_end_pred) & (stocks.symbol == sym)]

    future_feats.append({
        "symbol":        sym,
        "cur_weight":    cur_wt,
        "mom_30":        mom_30,
        "mom_60":        s60["log_return"].sum() if len(s60) >= 10 else mom_30,
        "mom_90":        s90["log_return"].sum() if len(s90) >= 15 else mom_30,
        "vol_30":        vol_30,
        "vol_60":        s60["log_return"].std() * np.sqrt(252) if len(s60) >= 10 else vol_30,
        "avg_volume":    avg_vol,
        "mkt_cap_proxy": last_price * avg_vol,
        "price_to_ma20": (last_price / s30["ma_20"].iloc[-1] - 1)
                          if "ma_20" in s30.columns and s30["ma_20"].iloc[-1] > 0 else 0,
        "price_to_ma50": (last_price / s30["ma_50"].iloc[-1] - 1)
                          if "ma_50" in s30.columns and s30["ma_50"].iloc[-1] > 0 else 0,
        "wt_drift":      s30["weight_pct"].iloc[-1] - s30["weight_pct"].iloc[0]
                          if len(s30) > 1 else 0,
        "wt_range":      s30["weight_pct"].max() - s30["weight_pct"].min()
                          if len(s30) > 1 else 0,
    })

future_df = pd.DataFrame(future_feats).fillna(train_medians)
X_future = scaler.transform(future_df[FEAT_COLS].values)

future_df["pred_weight_ridge"] = ridge.predict(X_future)
future_df["pred_weight_rf"]    = rf_reg.predict(X_future)
future_df["pred_retention_prob_logit"] = logit.predict_proba(X_future)[:, 1]
future_df["pred_retention_prob_rf"]    = rf_clf.predict_proba(X_future)[:, 1]
future_df["pred_retained_logit"] = logit.predict(X_future)
future_df["pred_retained_rf"]    = rf_clf.predict(X_future)

# Average retention probability
future_df["avg_retention_prob"] = (
    future_df["pred_retention_prob_logit"] +
    future_df["pred_retention_prob_rf"]
) / 2
future_df["risk_of_exclusion"] = 1 - future_df["avg_retention_prob"]

future_df = future_df.sort_values("risk_of_exclusion", ascending=False)

print(f"\nPredictions for 2025-09-15 rebalancing "
      f"({len(future_df)} current constituents):")
print(f"\n{'Symbol':8s} {'CurWt%':>7} {'PredWt(R)':>10} {'PredWt(RF)':>11} "
      f"{'RetProb':>9} {'Risk':>8}")
print("-" * 58)
for _, row in future_df.iterrows():
    flag = " ⚠ AT RISK" if row["risk_of_exclusion"] > 0.35 else ""
    print(f"{row['symbol']:8s} {row['cur_weight']:>7.2f} "
          f"{row['pred_weight_ridge']:>10.2f} "
          f"{row['pred_weight_rf']:>11.2f} "
          f"{row['avg_retention_prob']:>9.3f} "
          f"{row['risk_of_exclusion']:>8.3f}{flag}")

at_risk = future_df[future_df["risk_of_exclusion"] > 0.35]["symbol"].tolist()
print(f"\nStocks at risk of exclusion (>35% dropout probability): {at_risk}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: Generating figures ─────────────────────────────────")

# ── R1. Rebalancing composition history ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
# Track each symbol's presence by window
all_syms = sorted(panel["symbol"].unique())
n_syms   = len(all_syms)
n_wins   = len(REBAL_DATES) - 1
presence = np.zeros((n_syms, n_wins))
for w in range(n_wins):
    sub = panel[panel.win_idx == w]
    for s in sub[sub.retained == 1]["symbol"]:
        if s in all_syms:
            presence[all_syms.index(s), w] = 2  # retained
    for s in sub[sub.retained == 0]["symbol"]:
        if s in all_syms:
            presence[all_syms.index(s), w] = 1  # dropped this window
    # Stocks that enter (in next but not current) shown as entering
    if w < n_wins - 1:
        sub_next = panel[panel.win_idx == w + 1]
        next_syms = set(sub_next["symbol"])
        curr_syms_w = set(sub["symbol"])
        for s in next_syms - curr_syms_w:
            if s in all_syms:
                presence[all_syms.index(s), w] = 0.5  # entering next period

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#ecf0f1", "#f39c12", "#27ae60"])
window_labels = [f"{d.strftime('%b %Y')}" for d in REBAL_DATES[:-1]]
im = ax.imshow(presence, aspect="auto", cmap=cmap, vmin=0, vmax=2,
               interpolation="nearest")
ax.set_yticks(range(n_syms))
ax.set_yticklabels(all_syms, fontsize=7)
ax.set_xticks(range(n_wins))
ax.set_xticklabels(window_labels, rotation=45, ha="right", fontsize=8)
ax.set_title("KSE-30 Constituent Presence by Rebalancing Window\n"
             "(Green = retained, Orange = dropped, White = not in index)")
from matplotlib.patches import Patch
leg = [Patch(color="#27ae60", label="Retained"),
       Patch(color="#f39c12", label="Dropped"),
       Patch(color="#ecf0f1", label="Not in index")]
ax.legend(handles=leg, loc="upper right", fontsize=8)
# Mark train/test boundary
ax.axvline(max(TRAIN_WINDOWS) + 0.5, color="red", linewidth=2,
           linestyle="--", label="Train/test split")
ax.text(max(TRAIN_WINDOWS) + 0.6, 1, "TEST", color="red", fontsize=9, fontweight="bold")
plt.tight_layout()
savefig("R1_composition_history.png")

# ── R2. Weight prediction: actual vs predicted ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Task A: Weight Prediction — Actual vs Predicted (Test Set)",
             fontweight="bold")

# Keep only retained stocks for weight prediction plot
test_ret = test[test.retained == 1].copy()
test_ret["pred_ridge"] = ridge.predict(scaler.transform(test_ret[FEAT_COLS].values))
test_ret["pred_rf"]    = rf_reg.predict(scaler.transform(test_ret[FEAT_COLS].values))

for i, (model_name, pred_col) in enumerate([("Ridge regression","pred_ridge"),
                                              ("Random Forest",   "pred_rf")]):
    ax = axes[i]
    ax.scatter(test_ret["target_weight"], test_ret[pred_col],
               alpha=0.7, s=50, color=["#3498db","#e74c3c"][i])
    lim_max = max(test_ret["target_weight"].max(), test_ret[pred_col].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual weight (%)")
    ax.set_ylabel("Predicted weight (%)")
    m = reg_metrics(test_ret["target_weight"].values, test_ret[pred_col].values)
    ax.set_title(f"{model_name}\nRMSE={m['RMSE']:.3f}%  R²={m['R2']:.4f}")
    ax.legend(fontsize=8)
    # Annotate notable stocks
    for _, row in test_ret.nlargest(3, "target_weight").iterrows():
        ax.annotate(row["symbol"],
                    (row["target_weight"], row[pred_col]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.tight_layout()
savefig("R2_weight_prediction_scatter.png")

# ── R3. Feature importance ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Feature Importances — Weight & Retention Prediction", fontweight="bold")

ax = axes[0]
fi_reg = pd.DataFrame({"Feature": FEAT_COLS,
                        "Importance": rf_reg.feature_importances_})
fi_reg = fi_reg.sort_values("Importance")
ax.barh(fi_reg["Feature"], fi_reg["Importance"], color="#3498db", alpha=0.8)
ax.set_title("Random Forest — Weight Prediction\nFeature importance (MDI)")
ax.set_xlabel("Importance")

ax = axes[1]
fi_clf = pd.DataFrame({"Feature": FEAT_COLS,
                        "Importance": rf_clf.feature_importances_})
fi_clf = fi_clf.sort_values("Importance")
ax.barh(fi_clf["Feature"], fi_clf["Importance"], color="#e74c3c", alpha=0.8)
ax.set_title("Random Forest — Retention Prediction\nFeature importance (MDI)")
ax.set_xlabel("Importance")

plt.tight_layout()
savefig("R3_feature_importances.png")

# ── R4. Retention probability — current constituents (forward-looking) ────────
fig, ax = plt.subplots(figsize=(12, 6))
future_sorted = future_df.sort_values("avg_retention_prob")
colors = ["#e74c3c" if p < 0.65 else "#f39c12" if p < 0.80 else "#2ecc71"
          for p in future_sorted["avg_retention_prob"]]
bars = ax.barh(future_sorted["symbol"], future_sorted["avg_retention_prob"],
               color=colors, alpha=0.85)
ax.axvline(0.5, color="black", linewidth=1, linestyle="--", label="50% threshold")
ax.axvline(0.65, color="orange", linewidth=0.8, linestyle=":", label="65% threshold")
ax.set_xlabel("Predicted Retention Probability")
ax.set_title(f"Predicted Retention Probability for 2025-09-15 Rebalancing\n"
             f"(Ensemble of Logistic Regression + Random Forest)")
ax.legend(fontsize=8)
ax.set_xlim(0, 1.05)
for bar, val in zip(bars, future_sorted["avg_retention_prob"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8)
plt.tight_layout()
savefig("R4_retention_probability.png")

# ── R5. Predicted weight changes ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
future_wt = future_df.copy()
future_wt["wt_change_ridge"] = future_wt["pred_weight_ridge"] - future_wt["cur_weight"]
future_wt["wt_change_rf"]    = future_wt["pred_weight_rf"]    - future_wt["cur_weight"]
future_wt["avg_wt_change"]   = (future_wt["wt_change_ridge"] + future_wt["wt_change_rf"]) / 2
future_wt = future_wt.sort_values("avg_wt_change")

colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in future_wt["avg_wt_change"]]
ax.barh(future_wt["symbol"], future_wt["avg_wt_change"],
        color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Predicted Weight Change (%)")
ax.set_title("Predicted Weight Changes at 2025-09-15 Rebalancing\n"
             "(Ensemble average of Ridge + Random Forest)")
plt.tight_layout()
savefig("R5_predicted_weight_changes.png")

# ── R6. Confusion matrix for classification ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Task B: Inclusion Prediction — Confusion Matrices (Test Set)",
             fontweight="bold")
for i, (model_name, pred) in enumerate([("Logistic Regression", logit_pred_test),
                                          ("Random Forest",       rf_clf_pred_test)]):
    ax = axes[i]
    cm = confusion_matrix(y_ret_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dropped","Retained"],
                yticklabels=["Dropped","Retained"],
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    acc = accuracy_score(y_ret_test, pred)
    ax.set_title(f"{model_name}\nAccuracy = {acc:.3f}")
plt.tight_layout()
savefig("R6_confusion_matrices.png")

# ── R7. Feature distributions: retained vs dropped ────────────────────────────
key_feats = ["cur_weight","mom_30","vol_30","mkt_cap_proxy","wt_drift"]
fig, axes = plt.subplots(1, len(key_feats), figsize=(15, 4))
fig.suptitle("Feature Distributions: Retained vs Dropped Stocks", fontweight="bold")
for i, feat in enumerate(key_feats):
    ax = axes[i]
    ret_vals = panel[panel.retained == 1][feat].dropna()
    drp_vals = panel[panel.retained == 0][feat].dropna()
    ax.hist(ret_vals, bins=20, alpha=0.6, color="#2ecc71",
            density=True, label="Retained", edgecolor="white")
    ax.hist(drp_vals, bins=20, alpha=0.6, color="#e74c3c",
            density=True, label="Dropped",  edgecolor="white")
    ax.set_title(feat.replace("_"," "), fontsize=9)
    ax.set_xlabel("")
    if i == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
savefig("R7_feature_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY TABLE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 6: Summary results ────────────────────────────────────")

print("\n[Table A] Weight Prediction Results")
print(f"{'Model':30s} {'Train RMSE':>11} {'Test RMSE':>10} {'Test R²':>9}")
print("-" * 63)
for label, m_tr, m_te in [
    ("Naive (cur weight → next wt)", m_naive_wt_tr, m_naive_wt),
    ("Ridge regression",             m_ridge_tr,    m_ridge),
    ("Random Forest",                m_rf_tr,       m_rf),
]:
    print(f"{label:30s} {m_tr['RMSE']:>11.4f}% {m_te['RMSE']:>10.4f}% "
          f"{m_te['R2']:>9.4f}")

print("\n[Table B] Inclusion Prediction Results")
print(f"{'Model':30s} {'Train Acc':>10} {'Test Acc':>9} {'Test AUC':>9}")
print("-" * 60)
for label, m_tr, m_te in [
    ("Naive (all retained)",  {"Accuracy": y_ret_train.mean()}, m_naive_clf),
    ("Logistic regression",   m_logit_tr, m_logit),
    ("Random Forest",         m_rf_clf_tr, m_rf_clf),
]:
    print(f"{label:30s} {m_tr['Accuracy']:>10.3f} {m_te['Accuracy']:>9.3f} "
          f"{m_te.get('AUC', np.nan):>9.3f}")

print("\n[Table C] At-Risk Stocks — Sep 2025 Rebalancing Prediction")
at_risk_df = future_df[future_df["risk_of_exclusion"] > 0.20].sort_values(
    "risk_of_exclusion", ascending=False
)[["symbol","cur_weight","avg_retention_prob","risk_of_exclusion",
   "pred_weight_ridge","pred_weight_rf"]].round(4)
print(at_risk_df.to_string(index=False))

# Save results
results_rows = []
for label, m_tr, m_te, task in [
    ("Naive", m_naive_wt_tr, m_naive_wt, "Weight prediction"),
    ("Ridge regression", m_ridge_tr, m_ridge, "Weight prediction"),
    ("Random Forest", m_rf_tr, m_rf, "Weight prediction"),
]:
    results_rows.append({
        "Task": task, "Model": label,
        "Train RMSE": round(m_tr["RMSE"], 4),
        "Test RMSE":  round(m_te["RMSE"], 4),
        "Test MAE":   round(m_te["MAE"],  4),
        "Test R²":    round(m_te["R2"],   4),
    })
for label, m_tr, m_te in [
    ("Naive", {"Accuracy": float(y_ret_train.mean())}, m_naive_clf),
    ("Logistic regression", m_logit_tr, m_logit),
    ("Random Forest", m_rf_clf_tr, m_rf_clf),
]:
    results_rows.append({
        "Task": "Inclusion prediction", "Model": label,
        "Train Acc": round(m_tr["Accuracy"], 4),
        "Test Acc":  round(m_te["Accuracy"], 4),
        "Test AUC":  round(m_te.get("AUC", np.nan), 4),
    })

pd.DataFrame(results_rows).to_csv("results_rebalancing.csv", index=False)
print("\nSaved: results_rebalancing.csv")

future_df.to_csv("results_rebalancing_forecast.csv", index=False)
print("Saved: results_rebalancing_forecast.csv")

print(f"\nAll figures saved to: {FIG_DIR}")
print("REBALANCING PREDICTION COMPLETE.")
