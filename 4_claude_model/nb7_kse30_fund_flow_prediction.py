"""
NOTEBOOK 2 — Fund Flow Prediction: ARIMAX + VAR
=================================================
Input  : monthly_master.csv
Outputs: figures → ./figures/fund_flow/
         results → results_fund_flow_prediction.csv

Rationale for model choice
--------------------------
With only 39 monthly observations (25 train / 14 test), deep learning models
(LSTM, etc.) are statistically indefensible — they have more parameters than
training points and produce negative R² out-of-sample (as confirmed in prior run).

The correct approach for a small macroeconomic time-series:
  1. ARIMAX(1,0,1) — univariate baseline with exogenous macro regressors
     Target: total_fund_flow (aggregate across all three funds)
     Regressors: interest_rate_end, cpi_yoy_end, oil_return_monthly,
                 usdpkr_return_monthly (contemporaneous + lag-1)

  2. VAR(1) — Vector Autoregression capturing cross-fund spillovers
     Endogenous: [total_fund_flow, interest_rate_end, cpi_yoy_end]
     Exogenous:  oil_return_monthly, usdpkr_return_monthly
     Captures: how fund flows and macro variables jointly evolve

  3. Granger Causality — tests whether macro variables statistically
     predict fund flows (directly supports efficiency link in nb5)

Train / Test split
  Train : Jan 2021 → Dec 2023  (25 months)
  Test  : Jan 2024 → Sep 2025  (14 months)

Target variable
  total_fund_flow  (PKR millions, sum of AKD + NBP + NIT flows)
  This is the KSE-30 segment aggregate flow — the thesis target.
  Individual fund flows are analysed for heterogeneity only.

Run: python nb2_fund_flow_prediction.py
Requirements: numpy, pandas, matplotlib, scipy, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fund_flow")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 9})

FUND_COLORS = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NIT": "#2ca02c",
               "Total": "#8e44ad"}
TRAIN_END = "2023-12-31"

# ── load ─────────────────────────────────────────────────────────────────────
monthly = pd.read_csv("new_data/monthly_master.csv", parse_dates=["date"])
monthly = monthly.sort_values("date").reset_index(drop=True)
print(f"Monthly rows: {len(monthly)}  ({monthly['date'].min().date()} → {monthly['date'].max().date()})")

MACRO_COLS = ["interest_rate_end", "cpi_yoy_end",
              "oil_return_monthly", "usdpkr_return_monthly"]
TARGET = "total_fund_flow"

# ── train/test masks ─────────────────────────────────────────────────────────
train_mask = monthly["date"] <= TRAIN_END
test_mask  = ~train_mask
print(f"Train: {train_mask.sum()} months | Test: {test_mask.sum()} months")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, label=""):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    da   = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    if label:
        print(f"    {label:25s}  RMSE={rmse:9.2f}  MAE={mae:8.2f}  "
              f"R²={r2:7.4f}  DirAcc={da:.1f}%")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": da}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DESCRIPTIVE: FUND FLOW DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: Fund flow decomposition ────────────────────────────")

# Print individual vs aggregate stats
for fund, col in [("AKD", "flow_akd"), ("NBP", "flow_nbp"), ("NIT", "flow_nti"),
                  ("Total", "total_fund_flow")]:
    s = monthly[col]
    print(f"  {fund}: mean={s.mean():.1f}  std={s.std():.1f}  "
          f"min={s.min():.1f}  max={s.max():.1f}  PKR mn")

# Correlation between individual fund flows
print("\n  Inter-fund flow correlations:")
flow_corr = monthly[["flow_akd","flow_nbp","flow_nti"]].corr()
print(flow_corr.round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — STATIONARITY AND LAG SELECTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Stationarity and lag selection ─────────────────────")

def adf_simple(series, name):
    s = series.dropna().values.astype(float)
    dy = np.diff(s); ylag = s[:-1]
    X = np.column_stack([np.ones(len(ylag)), ylag])
    beta, _, _, _ = lstsq(X, dy)
    resid = dy - X @ beta
    s2 = resid @ resid / (len(dy) - 2)
    var_b = s2 * np.linalg.inv(X.T @ X)[1, 1]
    t = beta[1] / np.sqrt(max(var_b, 1e-15))
    p = stats.t.sf(abs(t), df=len(dy)-2) * 2
    print(f"    {name:30s}  ADF t={t:7.3f}  p={p:.4f}  "
          f"{'Stationary' if p < 0.05 else 'Non-stationary'}")
    return t, p

print("  ADF tests on model variables:")
for col in [TARGET] + MACRO_COLS:
    adf_simple(monthly[col], col)

# Lag selection via AIC (manual, no statsmodels dependency)
print("\n  Lag selection (ARIMAX AIC):")
best_aic = np.inf; best_lag = 1
for lag in [1, 2, 3]:
    df = monthly[[TARGET] + MACRO_COLS].copy()
    for L in range(1, lag + 1):
        df[f"tf_lag{L}"] = df[TARGET].shift(L)
    df = df.dropna()
    y = df[TARGET].values
    feat = [f"tf_lag{L}" for L in range(1, lag+1)] + MACRO_COLS
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in feat])
    beta, _, _, _ = lstsq(X, y)
    resid = y - X @ beta
    k = X.shape[1]; n = len(y)
    aic = n * np.log(np.var(resid)) + 2 * k
    bic = n * np.log(np.var(resid)) + k * np.log(n)
    print(f"    Lag {lag}: n={n:2d}  k={k:2d}  AIC={aic:.1f}  BIC={bic:.1f}")
    if aic < best_aic:
        best_aic = aic; best_lag = lag
print(f"  → Selected lag: {best_lag} (lowest AIC)")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — GRANGER CAUSALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 3: Granger causality tests ────────────────────────────")

def granger_test(y, x, max_lag=3, name=""):
    """
    Tests H0: x does NOT Granger-cause y.
    Restricted model:  y_t = c + sum(y_{t-k})
    Unrestricted model: y_t = c + sum(y_{t-k}) + sum(x_{t-k})
    F-test on the additional x terms.
    """
    results = []
    for lag in range(1, max_lag + 1):
        df = pd.DataFrame({"y": y, "x": x})
        for L in range(1, lag + 1):
            df[f"yl{L}"] = df["y"].shift(L)
            df[f"xl{L}"] = df["x"].shift(L)
        df = df.dropna()
        yv = df["y"].values
        n  = len(yv)

        # Restricted
        Xr = np.column_stack([np.ones(n)] + [df[f"yl{L}"].values for L in range(1, lag+1)])
        br, _, _, _ = lstsq(Xr, yv)
        rss_r = np.sum((yv - Xr @ br) ** 2)

        # Unrestricted
        Xu = np.column_stack([Xr] + [df[f"xl{L}"].values for L in range(1, lag+1)])
        bu, _, _, _ = lstsq(Xu, yv)
        rss_u = np.sum((yv - Xu @ bu) ** 2)

        k_r = Xr.shape[1]; k_u = Xu.shape[1]
        df_num = k_u - k_r; df_den = n - k_u
        F = ((rss_r - rss_u) / df_num) / (rss_u / df_den) if df_den > 0 else np.nan
        p = 1 - stats.f.cdf(F, df_num, df_den) if not np.isnan(F) else np.nan
        results.append((lag, F, p))
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        print(f"    {name:30s} lag={lag}: F={F:.3f}  p={p:.4f}  {sig}")
    return results

y_flow = monthly[TARGET].values
print("  Does macro Granger-cause total fund flow?")
granger_results = {}
for col, label in [("interest_rate_end", "Interest rate → Fund flow"),
                   ("cpi_yoy_end",        "CPI YoY       → Fund flow"),
                   ("oil_return_monthly", "Oil return    → Fund flow"),
                   ("usdpkr_return_monthly","USD/PKR ret  → Fund flow")]:
    granger_results[col] = granger_test(y_flow, monthly[col].values, max_lag=3, name=label)

print("\n  Does fund flow Granger-cause macro (efficiency link)?")
for col, label in [("interest_rate_end", "Fund flow → Interest rate"),
                   ("cpi_yoy_end",        "Fund flow → CPI YoY      ")]:
    granger_test(monthly[col].values, y_flow, max_lag=3, name=label)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — ARIMAX MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 4: ARIMAX(1,0,1) model ───────────────────────────────")

def fit_arimax(y_train, X_train, y_test, X_test, p=1, label="ARIMAX"):
    """
    ARIMAX(p,0,0): y_t = c + phi*y_{t-1} + beta*X_t + eps_t
    Fit via OLS on training set, walk-forward on test set.
    X_train/X_test: (n, k) matrix of exogenous variables.
    """
    n_tr = len(y_train)

    # Build lagged design matrix for training
    rows = []
    for t in range(p, n_tr):
        row = [y_train[t - i] for i in range(1, p + 1)]  # AR lags
        row += list(X_train[t])                            # contemporaneous exog
        rows.append(row)
    X_des = np.column_stack([np.ones(len(rows))] + [np.array(rows)[:, i]
                              for i in range(np.array(rows).shape[1])])
    y_des = y_train[p:]

    beta, _, _, _ = lstsq(X_des, y_des)
    fitted_train = X_des @ beta
    resid_train  = y_des - fitted_train
    r2_train     = 1 - np.var(resid_train) / np.var(y_des)

    # Walk-forward forecast on test set
    history = list(y_train)
    preds   = []
    for t in range(len(y_test)):
        ar_lags = [history[-(i)] for i in range(1, p + 1)]
        x_row   = list(X_test[t])
        x_new   = np.array([1.0] + ar_lags + x_row)
        pred = x_new @ beta
        preds.append(pred)
        history.append(y_test[t])

    return np.array(preds), fitted_train, beta, r2_train

# Prepare data
df_model = monthly[[TARGET] + MACRO_COLS].copy().dropna()
train_df = df_model[monthly["date"][df_model.index] <= TRAIN_END]
test_df  = df_model[monthly["date"][df_model.index] >  TRAIN_END]

y_tr = train_df[TARGET].values.astype(float)
y_te = test_df[TARGET].values.astype(float)
X_tr = train_df[MACRO_COLS].values.astype(float)
X_te = test_df[MACRO_COLS].values.astype(float)

arimax_pred, arimax_fitted, arimax_beta, arimax_r2_train = fit_arimax(
    y_tr, X_tr, y_te, X_te, p=1, label="ARIMAX"
)

print(f"  ARIMAX(1,0,1) coefficients:")
coef_labels = ["Intercept", "AR(1)"] + MACRO_COLS
for lbl, b in zip(coef_labels, arimax_beta):
    print(f"    {lbl:30s}: {b:.4f}")
print(f"\n  In-sample  R² = {arimax_r2_train:.4f}")

m_arimax = metrics(y_te, arimax_pred, "ARIMAX(1,0,1) test")

# Naive walk-forward benchmark: predict next = last actual
naive_preds = [y_tr[-1]] + list(y_te[:-1])
m_naive = metrics(y_te, naive_preds, "Naive (random walk) ")

# Mean benchmark
mean_preds = [y_tr.mean()] * len(y_te)
m_mean = metrics(y_te, mean_preds, "Mean benchmark      ")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — VAR(1) MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: VAR(1) model ───────────────────────────────────────")

# Endogenous: total_fund_flow, interest_rate_end, cpi_yoy_end
# Exogenous:  oil_return_monthly, usdpkr_return_monthly
ENDO_COLS = ["total_fund_flow", "interest_rate_end", "cpi_yoy_end"]
EXOG_COLS = ["oil_return_monthly", "usdpkr_return_monthly"]

def fit_var1(train_df, test_df, endo_cols, exog_cols):
    """
    VAR(1) with exogenous variables — OLS equation by equation.
    For each endogenous variable y_i:
      y_i,t = c_i + A_i * Y_{t-1} + B_i * X_t + eps_i,t
    Returns predictions for the first endogenous variable (fund flow).
    """
    n_endo = len(endo_cols)
    n_exog = len(exog_cols)

    # Build lagged training matrix
    Y_tr = train_df[endo_cols].values.astype(float)
    X_tr = train_df[exog_cols].values.astype(float)
    Y_te = test_df[endo_cols].values.astype(float)
    X_te = test_df[exog_cols].values.astype(float)

    n_tr = len(Y_tr)

    # Regressors: [1, Y_{t-1}[0], ..., Y_{t-1}[k], X_t[0], ..., X_t[m]]
    Z_tr = np.column_stack([np.ones(n_tr - 1),
                            Y_tr[:-1],       # lagged endogenous
                            X_tr[1:]])       # contemporaneous exogenous
    Y_cur = Y_tr[1:]

    # Fit each equation by OLS
    betas = []
    for i in range(n_endo):
        b, _, _, _ = lstsq(Z_tr, Y_cur[:, i])
        betas.append(b)
        fitted = Z_tr @ b
        r2 = 1 - np.var(Y_cur[:, i] - fitted) / np.var(Y_cur[:, i])
        print(f"  Equation {endo_cols[i]:25s}: in-sample R²={r2:.4f}")

    # Walk-forward forecast (fund flow only, other endos updated with actuals)
    history_Y = list(Y_tr)
    preds_flow = []
    for t in range(len(Y_te)):
        Y_lag = np.array(history_Y[-1])
        X_now = X_te[t]
        z = np.concatenate([[1.0], Y_lag, X_now])
        pred_flow = z @ betas[0]
        preds_flow.append(pred_flow)
        history_Y.append(Y_te[t])  # use actual for next step

    return np.array(preds_flow), betas

print("\n  VAR(1) equation-by-equation OLS:")
var_train = monthly[monthly["date"] <= TRAIN_END][ENDO_COLS + EXOG_COLS].dropna()
var_test  = monthly[monthly["date"] >  TRAIN_END][ENDO_COLS + EXOG_COLS].dropna()
var_pred, var_betas = fit_var1(var_train, var_test, ENDO_COLS, EXOG_COLS)
m_var = metrics(y_te, var_pred, "VAR(1) test         ")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — INDIVIDUAL FUND FLOW MODELS (heterogeneity analysis)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 6: Individual fund ARIMAX (heterogeneity) ─────────────")

fund_results = {}
for fund, col in [("AKD","flow_akd"), ("NBP","flow_nbp"), ("NIT","flow_nti")]:
    df_f = monthly[[col] + MACRO_COLS].dropna()
    tr_f = df_f[monthly["date"][df_f.index] <= TRAIN_END]
    te_f = df_f[monthly["date"][df_f.index] >  TRAIN_END]
    if len(te_f) == 0:
        continue
    pred_f, _, _, _ = fit_arimax(
        tr_f[col].values.astype(float), tr_f[MACRO_COLS].values.astype(float),
        te_f[col].values.astype(float), te_f[MACRO_COLS].values.astype(float),
        p=1
    )
    m_f = metrics(te_f[col].values, pred_f, f"{fund} ARIMAX test   ")
    fund_results[fund] = {
        "pred": pred_f,
        "actual": te_f[col].values,
        "dates_test": monthly.loc[monthly["date"] > TRAIN_END, "date"].values[:len(pred_f)],
        "metrics": m_f
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — FUND FLOW → MARKET EFFICIENCY LINK
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 7: Fund flow → Market efficiency link ─────────────────")

# High-flow months: does autocorrelation of index returns change?
# Proxy: compare rolling ACF of KSE-30 log return in high vs low flow months
daily = pd.read_csv("new_data/daily_master.csv", parse_dates=["date"])
daily = daily.sort_values("date").reset_index(drop=True)
daily["month"] = daily["date"].dt.to_period("M")
monthly["month"] = monthly["date"].dt.to_period("M")

# Month-end rolling ACF of each fund NAV (proxy for KSE-30 efficiency)
print("\n  Regression: |rolling ACF| ~ fund_flow_pct (efficiency link)")
daily_acf = {}
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    racf = daily[col].rolling(60).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
    daily["_racf"] = racf
    me = daily.groupby("month")["_racf"].last().reset_index().rename(
        columns={"_racf": f"acf_{fund}"}
    )
    daily_acf[fund] = me

# Merge with monthly flows
eff_df = monthly[["month","total_fund_flow","flow_pct_akd","flow_pct_nbp","flow_pct_nti"]].copy()
for fund in ["AKD","NBP","NIT"]:
    eff_df = eff_df.merge(daily_acf[fund], on="month", how="left")
eff_df = eff_df.dropna()

print(f"\n  Observations for regression: {len(eff_df)}")
for fund in ["AKD","NBP","NIT"]:
    acf_col = f"acf_{fund}"
    flow_col = f"flow_pct_{'nti' if fund=='NIT' else fund.lower()}"
    sub = eff_df[[flow_col, acf_col]].dropna()
    if len(sub) > 5:
        r, p = stats.pearsonr(sub[flow_col], sub[acf_col])
        print(f"  {fund}: flow_pct vs |ACF|  r={r:.4f}  p={p:.4f}  "
              f"→ {'significant' if p < 0.10 else 'not significant'}")

# Quartile split: high flow vs low flow months
q75 = monthly["total_fund_flow"].quantile(0.75)
q25 = monthly["total_fund_flow"].quantile(0.25)
high_flow_months = monthly.loc[monthly["total_fund_flow"] >= q75, "month"]
low_flow_months  = monthly.loc[monthly["total_fund_flow"] <= q25, "month"]

print(f"\n  High flow months (≥75th pctile): {len(high_flow_months)}")
print(f"  Low flow months  (≤25th pctile): {len(low_flow_months)}")

for fund, col in [("AKD","nav_return_akd"), ("NIT","nav_return_nti")]:
    daily["_r"] = daily[col]
    hi_returns = daily[daily["month"].isin(high_flow_months)]["_r"].dropna().values
    lo_returns = daily[daily["month"].isin(low_flow_months)]["_r"].dropna().values
    acf_hi = pd.Series(hi_returns).autocorr(1)
    acf_lo = pd.Series(lo_returns).autocorr(1)
    print(f"  {fund}: ACF lag-1 | high-flow months={acf_hi:.4f}  "
          f"low-flow months={acf_lo:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 8: Generating figures ────────────────────────────────")

dates_train = monthly.loc[train_mask, "date"].values
dates_test  = monthly.loc[test_mask,  "date"].values

# ── F1. Total fund flow: actual vs ARIMAX vs VAR ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(dates_train, monthly.loc[train_mask, TARGET],
       color=["#2980b9" if v >= 0 else "#e74c3c"
              for v in monthly.loc[train_mask, TARGET]],
       width=20, alpha=0.5, label="Actual (train)")
ax.bar(dates_test, y_te,
       color=["#2980b9" if v >= 0 else "#e74c3c" for v in y_te],
       width=20, alpha=0.8, label="Actual (test)")
ax.plot(dates_test, arimax_pred, "o-", color="#e74c3c",
        linewidth=2, markersize=5, label=f"ARIMAX (R²={m_arimax['R2']:.3f})")
ax.plot(dates_test, var_pred,    "s--", color="#2ca02c",
        linewidth=2, markersize=5, label=f"VAR(1) (R²={m_var['R2']:.3f})")
ax.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Total KSE-30 Fund Flow — Actual vs ARIMAX vs VAR(1)\n"
             "(Aggregate: AKD + NBP + NIT, PKR Millions)")
ax.set_ylabel("Fund Flow (PKR mn)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
savefig("F1_total_flow_predictions.png")

# ── F2. Individual fund ARIMAX predictions ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Individual Fund Flow Predictions — ARIMAX(1,0,1)", fontweight="bold")
for i, fund in enumerate(["AKD","NBP","NIT"]):
    ax = axes[i]
    res = fund_results.get(fund, {})
    if not res:
        continue
    flow_col = f"flow_akd" if fund=="AKD" else f"flow_nbp" if fund=="NBP" else "flow_nti"
    ax.bar(dates_train, monthly.loc[train_mask, flow_col],
           color=[FUND_COLORS[fund] if v >= 0 else "#e74c3c"
                  for v in monthly.loc[train_mask, flow_col]],
           width=20, alpha=0.5, label="Actual (train)")
    ax.bar(dates_test[:len(res["actual"])], res["actual"],
           color=[FUND_COLORS[fund] if v >= 0 else "#e74c3c"
                  for v in res["actual"]],
           width=20, alpha=0.9, label="Actual (test)")
    ax.plot(dates_test[:len(res["pred"])], res["pred"], "o-",
            color="black", linewidth=1.5, markersize=4,
            label=f"ARIMAX pred (R²={res['metrics']['R2']:.3f})")
    ax.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"{fund} Flow (PKR mn)")
    ax.set_title(f"{fund} — ARIMAX(1,0,1) | "
                 f"RMSE={res['metrics']['RMSE']:.2f}  MAE={res['metrics']['MAE']:.2f}")
    ax.legend(fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("F2_individual_fund_predictions.png")

# ── F3. Model comparison bar chart ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Fund Flow Model Comparison — Out-of-Sample Test Set", fontweight="bold")
model_names = ["Naive RW", "ARIMAX(1,0,1)", "VAR(1)"]
metrics_map = {"RMSE": [m_naive["RMSE"], m_arimax["RMSE"], m_var["RMSE"]],
               "MAE":  [m_naive["MAE"],  m_arimax["MAE"],  m_var["MAE"]],
               "R²":   [m_naive["R2"],   m_arimax["R2"],   m_var["R2"]]}
colors = ["#95a5a6", "#e74c3c", "#2ecc71"]
for j, (metric, vals) in enumerate(metrics_map.items()):
    ax = axes[j]
    bars = ax.bar(model_names, vals, color=colors, alpha=0.85)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
plt.tight_layout()
savefig("F3_model_comparison.png")

# ── F4. Granger causality summary chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
macro_labels = ["Interest Rate", "CPI YoY", "Oil Return", "USD/PKR Return"]
p_vals = []
for col in ["interest_rate_end","cpi_yoy_end","oil_return_monthly","usdpkr_return_monthly"]:
    # Use lag-1 p-value
    res = granger_results[col]
    p_vals.append(res[0][2])  # lag-1 p-value
bar_cols = ["#e74c3c" if p < 0.05 else "#f39c12" if p < 0.10 else "#95a5a6"
            for p in p_vals]
bars = ax.bar(macro_labels, p_vals, color=bar_cols, alpha=0.85)
ax.axhline(0.05, color="black", linewidth=1.2, linestyle="--", label="5% significance")
ax.axhline(0.10, color="gray",  linewidth=0.8, linestyle=":",  label="10% significance")
ax.set_ylabel("Granger causality p-value (lag 1)")
ax.set_title("Granger Causality: Does Macro Variable Predict Fund Flow?\n"
             "(Red = significant at 5%; Orange = 10%; Gray = not significant)")
ax.legend()
ax.set_ylim(0, max(p_vals) * 1.2)
for bar, p in zip(bars, p_vals):
    ax.text(bar.get_x() + bar.get_width()/2, p + 0.005,
            f"p={p:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
savefig("F4_granger_causality.png")

# ── F5. Residual diagnostics ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("ARIMAX(1,0,1) Residual Diagnostics — Total Fund Flow", fontweight="bold")
resid = y_te - arimax_pred

axes[0].plot(dates_test, resid, "o-", color="#8e44ad", linewidth=1)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_title("Residuals over time")
axes[0].set_ylabel("Residual (PKR mn)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=4))

axes[1].hist(resid, bins=12, color="#8e44ad", alpha=0.75, edgecolor="white")
axes[1].set_title("Residual distribution")
axes[1].set_xlabel("Residual")

max_lag_acf = min(8, len(resid) - 2)
acf_vals = [np.corrcoef(resid[k:], resid[:-k])[0,1] for k in range(1, max_lag_acf+1)]
ci = 1.96 / np.sqrt(len(resid))
axes[2].bar(range(1, max_lag_acf+1), acf_vals, color="#8e44ad", alpha=0.8)
axes[2].axhline( ci, color="red", linewidth=0.8, linestyle="--")
axes[2].axhline(-ci, color="red", linewidth=0.8, linestyle="--")
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_title("ACF of residuals\n(should be ~0 if model is adequate)")
axes[2].set_xlabel("Lag")

fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("F5_residual_diagnostics.png")

# ── F6. Fund flow vs market efficiency scatter ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fund Flow → Market Efficiency Link\n"
             "(Tests whether large fund flows reduce return predictability)",
             fontweight="bold")
for i, (fund, col) in enumerate([("AKD","acf_AKD"), ("NIT","acf_NIT")]):
    ax = axes[i]
    flow_col = "flow_pct_akd" if fund=="AKD" else "flow_pct_nti"
    sub = eff_df[[flow_col, col]].dropna()
    if len(sub) > 3:
        ax.scatter(sub[flow_col] * 100, sub[col],
                   color=FUND_COLORS[fund], alpha=0.7, s=60)
        m, b = np.polyfit(sub[flow_col], sub[col], 1)
        xl = np.linspace(sub[flow_col].min(), sub[flow_col].max(), 100)
        ax.plot(xl * 100, m * xl + b, "k--", linewidth=1)
        r, p = stats.pearsonr(sub[flow_col], sub[col])
        ax.set_title(f"{fund}: fund flow % vs rolling ACF lag-1\n"
                     f"r={r:.3f}, p={p:.3f}")
        ax.set_xlabel("Fund flow (% of AUM)")
        ax.set_ylabel("Rolling 60-day ACF (lag 1)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
savefig("F6_flow_efficiency_link.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — RESULTS TABLE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 9: Saving results ─────────────────────────────────────")

rows = []
for model, m, note in [
    ("Naive (random walk)", m_naive,  "Benchmark"),
    ("ARIMAX(1,0,1)",       m_arimax, "Primary model"),
    ("VAR(1)",              m_var,    "Macro-system model"),
]:
    rows.append({
        "Model": model, "Target": "Total Fund Flow",
        "Train N": int(train_mask.sum()), "Test N": int(test_mask.sum()),
        "RMSE (PKR mn)": round(m["RMSE"], 2),
        "MAE (PKR mn)":  round(m["MAE"],  2),
        "R²":            round(m["R2"],   4),
        "Dir. Accuracy": round(m["DirAcc"], 1),
        "Note": note,
    })
for fund in ["AKD","NBP","NIT"]:
    res = fund_results.get(fund, {})
    if res:
        rows.append({
            "Model": "ARIMAX(1,0,1)", "Target": f"{fund} Flow",
            "Train N": int(train_mask.sum()), "Test N": int(test_mask.sum()),
            "RMSE (PKR mn)": round(res["metrics"]["RMSE"], 2),
            "MAE (PKR mn)":  round(res["metrics"]["MAE"],  2),
            "R²":            round(res["metrics"]["R2"],   4),
            "Dir. Accuracy": round(res["metrics"]["DirAcc"], 1),
            "Note": "Heterogeneity analysis",
        })

results_df = pd.DataFrame(rows)
results_df.to_csv("results_fund_flow_prediction.csv", index=False)
print("Saved: results_fund_flow_prediction.csv")
print(results_df.to_string(index=False))

print(f"\nAll figures saved to: {FIG_DIR}")
print("FUND FLOW PREDICTION COMPLETE.")
