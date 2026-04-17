"""
NOTEBOOK 5 — Market Efficiency Analysis
=========================================
Inputs : daily_master.csv, monthly_master.csv, kse30_stocks_daily.csv
Outputs: figures → ./figures/efficiency/
         results → printed to console (copy to thesis)

Tests implemented
  1. Runs Test          — non-parametric, tests for randomness of return signs
  2. Variance Ratio Test — Lo & MacKinlay (1988), tests random-walk hypothesis
  3. Autocorrelation Test — Ljung-Box Q on index returns
  4. Hurst Exponent      — long-memory measure (H=0.5 → random walk)
  5. Fund Flow → Efficiency Link — does high fund flow predict return
                                    autocorrelation (market inefficiency)?

Scope: applied to each fund's daily NAV return series as proxy for
       KSE-30 segment efficiency (since kse30_index_level not in daily_master
       at this stage; fund NAVs reflect the same underlying market).
       Also applied stock-by-stock for the 19 active constituents.

Run: python nb5_market_efficiency.py
Requirements: numpy, pandas, matplotlib, seaborn, scipy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "efficiency")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 9})

FUND_COLORS = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NIT": "#2ca02c"}

# ── load ─────────────────────────────────────────────────────────────────────
daily   = pd.read_csv("daily_master.csv",           parse_dates=["date"])
monthly = pd.read_csv("monthly_master.csv",         parse_dates=["date"])
stocks  = pd.read_csv("kse30_stocks_daily.csv",     parse_dates=["date"])
daily   = daily.sort_values("date").reset_index(drop=True)
monthly = monthly.sort_values("date").reset_index(drop=True)

print(f"Daily  : {len(daily):,} rows  {daily['date'].min().date()} → {daily['date'].max().date()}")
print(f"Monthly: {len(monthly):,} rows")


# ═══════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def runs_test(returns, name=""):
    """
    Wald-Wolfowitz Runs Test for randomness.
    H0: Returns are independently and randomly distributed (weak-form efficient).
    A 'run' is a sequence of consecutive positive or negative returns.

    Returns: (Z-statistic, p-value, n_runs, expected_runs, verdict)
    """
    r = np.array(returns)
    r = r[~np.isnan(r)]
    signs = np.where(r >= 0, 1, -1)

    n_pos = np.sum(signs == 1)
    n_neg = np.sum(signs == -1)
    n     = n_pos + n_neg

    # Count runs
    n_runs = 1 + np.sum(signs[1:] != signs[:-1])

    # Expected runs and variance under H0
    expected = (2 * n_pos * n_neg) / n + 1
    variance = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

    if variance <= 0:
        return None

    Z   = (n_runs - expected) / np.sqrt(variance)
    p   = 2 * (1 - stats.norm.cdf(abs(Z)))
    verdict = "EFFICIENT (H0 not rejected)" if p >= 0.05 else "INEFFICIENT (H0 rejected)"

    return {
        "name":           name,
        "n":              n,
        "n_pos":          n_pos,
        "n_neg":          n_neg,
        "n_runs":         int(n_runs),
        "expected_runs":  round(expected, 2),
        "Z":              round(Z, 4),
        "p_value":        round(p, 4),
        "verdict":        verdict,
    }


def variance_ratio_test(returns, q=2, name=""):
    """
    Lo-MacKinlay (1988) Variance Ratio Test.
    H0: Return series follows a random walk (VR = 1).
    VR(q) = Var(q-period return) / (q × Var(1-period return))
    Under RW: VR = 1. VR > 1 → positive autocorrelation. VR < 1 → mean reversion.

    Uses heteroskedasticity-robust Z* statistic.
    """
    r = np.array(returns)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < q * 2:
        return None

    # q-period returns
    r_q = np.array([np.sum(r[i:i+q]) for i in range(T - q + 1)])

    sigma1 = np.var(r,   ddof=1)
    sigma_q = np.var(r_q, ddof=1) / q

    VR = sigma_q / sigma1 if sigma1 > 0 else np.nan

    # Heteroskedasticity-robust delta
    mu  = np.mean(r)
    numer = np.sum((r - mu)**2) ** 2
    delta = np.zeros(q - 1)
    for k in range(1, q):
        num_k = np.sum((r[k:] - mu)**2 * (r[:-k] - mu)**2)
        delta[k-1] = num_k / numer * T

    theta = 4 * np.sum([(1 - k/q)**2 * delta[k-1] for k in range(1, q)])
    Z_star = (VR - 1) / np.sqrt(theta / T) if theta > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(Z_star))) if not np.isnan(Z_star) else np.nan
    verdict = "EFFICIENT (RW not rejected)" if p >= 0.05 else "INEFFICIENT (RW rejected)"

    return {
        "name":    name,
        "q":       q,
        "T":       T,
        "VR":      round(VR, 4),
        "Z_star":  round(Z_star, 4) if not np.isnan(Z_star) else np.nan,
        "p_value": round(p, 4) if not np.isnan(p) else np.nan,
        "verdict": verdict,
    }


def autocorrelation_test(returns, lags=10, name=""):
    """
    Ljung-Box Q test for serial autocorrelation.
    H0: No autocorrelation up to lag k (consistent with random walk).
    """
    r = np.array(returns)
    r = r[~np.isnan(r)]
    n = len(r)
    mu = r.mean()

    acf_vals = []
    for k in range(1, lags + 1):
        num = np.sum((r[k:] - mu) * (r[:-k] - mu))
        den = np.sum((r - mu)**2)
        acf_vals.append(num / den if den > 0 else 0)

    Q = n * (n + 2) * sum(rk**2 / (n - k)
                          for k, rk in enumerate(acf_vals, 1))
    p = 1 - stats.chi2.cdf(Q, df=lags)
    verdict = "EFFICIENT (no autocorr)" if p >= 0.05 else "INEFFICIENT (autocorr detected)"
    return {
        "name":    name,
        "lags":    lags,
        "Q":       round(Q, 4),
        "p_value": round(p, 4),
        "acf":     [round(a, 4) for a in acf_vals],
        "verdict": verdict,
    }


def hurst_exponent(returns, name=""):
    """
    R/S Analysis — Hurst Exponent.
    H = 0.5 → random walk (efficient)
    H > 0.5 → persistent (trending, mean momentum)
    H < 0.5 → anti-persistent (mean-reverting)
    """
    r = np.array(returns)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 20:
        return None

    lags = range(2, min(n // 4, 100))
    RS   = []
    for lag in lags:
        # Slice into non-overlapping windows
        sub_series = [r[i:i+lag] for i in range(0, n - lag + 1, lag)]
        sub_RS = []
        for sub in sub_series:
            if len(sub) < 2:
                continue
            mean_sub  = np.mean(sub)
            devs      = np.cumsum(sub - mean_sub)
            R_val     = devs.max() - devs.min()
            S_val     = np.std(sub, ddof=1)
            if S_val > 0:
                sub_RS.append(R_val / S_val)
        if sub_RS:
            RS.append(np.mean(sub_RS))
        else:
            RS.append(np.nan)

    valid = [(lag, rs) for lag, rs in zip(lags, RS) if not np.isnan(rs) and rs > 0]
    if len(valid) < 5:
        return None

    log_lags = np.log([v[0] for v in valid])
    log_rs   = np.log([v[1] for v in valid])
    H, _, r2, _, _ = stats.linregress(log_lags, log_rs)

    if   H < 0.45: interp = "Anti-persistent (mean-reverting)"
    elif H > 0.55: interp = "Persistent (trending / long memory)"
    else:          interp = "Random walk (H ≈ 0.5)"

    return {
        "name":          name,
        "H":             round(H, 4),
        "r2":            round(r2**2, 4) if not np.isnan(r2) else np.nan,
        "interpretation": interp,
    }


def rolling_autocorrelation(returns, dates, window=60, lag=1):
    """
    Rolling lag-1 autocorrelation over a window.
    High autocorrelation → inefficiency. Near zero → efficiency.
    """
    r = pd.Series(returns, index=dates)
    roll_acf = r.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )
    return roll_acf


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — FUND NAV RETURN EFFICIENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: Fund NAV efficiency tests ──────────────────────────")

fund_series = {
    "AKD": daily["nav_return_akd"].dropna().values,
    "NBP": daily["nav_return_nbp"].dropna().values,
    "NIT": daily["nav_return_nti"].dropna().values,
}
fund_dates = {
    "AKD": daily.loc[daily["nav_return_akd"].notna(), "date"].values,
    "NBP": daily.loc[daily["nav_return_nbp"].notna(), "date"].values,
    "NIT": daily.loc[daily["nav_return_nti"].notna(), "date"].values,
}

runs_results   = {}
vr_results     = {}
acf_results    = {}
hurst_results  = {}

for fund, r in fund_series.items():
    print(f"\n  ── {fund} (n={len(r)}) ──")

    rt = runs_test(r, name=fund)
    runs_results[fund] = rt
    print(f"  Runs Test     : Z={rt['Z']:>8.4f}  p={rt['p_value']:.4f}  "
          f"runs={rt['n_runs']} (exp={rt['expected_runs']})  → {rt['verdict']}")

    for q in [2, 4, 8, 16]:
        vr = variance_ratio_test(r, q=q, name=fund)
        if fund not in vr_results:
            vr_results[fund] = []
        vr_results[fund].append(vr)
        print(f"  VR Test (q={q:2d}): VR={vr['VR']:.4f}  Z*={vr['Z_star']:>8.4f}  "
              f"p={vr['p_value']:.4f}  → {vr['verdict']}")

    acf = autocorrelation_test(r, lags=10, name=fund)
    acf_results[fund] = acf
    print(f"  Ljung-Box Q   : Q={acf['Q']:.4f}  p={acf['p_value']:.4f}  → {acf['verdict']}")
    print(f"  ACF lags 1-5  : {acf['acf'][:5]}")

    H = hurst_exponent(r, name=fund)
    hurst_results[fund] = H
    if H:
        print(f"  Hurst Exponent: H={H['H']:.4f}  → {H['interpretation']}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — STOCK-LEVEL EFFICIENCY TESTS (Variance Ratio q=2)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Stock-level efficiency tests ───────────────────────")

active_syms = (
    stocks.groupby("symbol")["date"].count()
    [lambda s: s >= 0.80 * stocks["date"].nunique()]
    .index.tolist()
)
returns_wide = (
    stocks[stocks["symbol"].isin(active_syms)]
    .pivot_table(index="date", columns="symbol", values="log_return")
    .sort_index()
    .ffill(limit=2)
    .fillna(0)
)

stock_eff = []
for sym in returns_wide.columns:
    r = returns_wide[sym].dropna().values
    rt  = runs_test(r, name=sym)
    vr2 = variance_ratio_test(r, q=2, name=sym)
    acf = autocorrelation_test(r, lags=5, name=sym)
    H   = hurst_exponent(r, name=sym)
    stock_eff.append({
        "Symbol":        sym,
        "Runs Z":        rt["Z"] if rt else np.nan,
        "Runs p":        rt["p_value"] if rt else np.nan,
        "Runs verdict":  "Inefficient" if rt and rt["p_value"] < 0.05 else "Efficient",
        "VR(2)":         vr2["VR"] if vr2 else np.nan,
        "VR p":          vr2["p_value"] if vr2 else np.nan,
        "VR verdict":    "Inefficient" if vr2 and vr2["p_value"] < 0.05 else "Efficient",
        "LB Q p":        acf["p_value"] if acf else np.nan,
        "LB verdict":    "Inefficient" if acf and acf["p_value"] < 0.05 else "Efficient",
        "Hurst H":       H["H"] if H else np.nan,
        "Hurst interp":  H["interpretation"] if H else "",
    })

eff_df = pd.DataFrame(stock_eff).set_index("Symbol")
print("\n[Table] Stock-Level Efficiency Test Results:")
print(eff_df[["Runs Z", "Runs p", "Runs verdict",
              "VR(2)", "VR p", "VR verdict",
              "LB Q p", "Hurst H"]].round(4).to_string())

# Summary counts
n_ineff_runs = (eff_df["Runs p"] < 0.05).sum()
n_ineff_vr   = (eff_df["VR p"]   < 0.05).sum()
n_ineff_lb   = (eff_df["LB Q p"] < 0.05).sum()
n_total = len(eff_df)
print(f"\nInefficiency rates across {n_total} stocks:")
print(f"  Runs Test   : {n_ineff_runs}/{n_total} stocks inefficient ({n_ineff_runs/n_total*100:.0f}%)")
print(f"  VR Test q=2 : {n_ineff_vr}/{n_total} stocks inefficient ({n_ineff_vr/n_total*100:.0f}%)")
print(f"  Ljung-Box Q : {n_ineff_lb}/{n_total} stocks inefficient ({n_ineff_lb/n_total*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUND FLOW → EFFICIENCY LINK
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 3: Fund flow vs market efficiency link ────────────────")

# Rolling 60-day lag-1 autocorrelation of each fund's NAV return
# → use as a proxy for time-varying inefficiency
# Then correlate with monthly fund flows

rolling_acf = {}
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r = daily[col]
    dt = daily["date"]
    racf = rolling_autocorrelation(r, dt, window=60, lag=1)
    rolling_acf[fund] = racf

# Merge rolling ACF (daily) with monthly fund flows by month-end
daily["month"] = daily["date"].dt.to_period("M")
monthly["month"] = monthly["date"].dt.to_period("M")

flow_acf_corrs = {}
for fund in ["AKD", "NBP", "NIT"]:
    col_flow = f"flow_pct_{'nti' if fund=='NIT' else fund.lower()}"
    # Month-end rolling ACF
    racf_series = rolling_acf[fund].reset_index(drop=True)
    daily["racf_tmp"] = racf_series.values
    me_acf = (
        daily.groupby("month")["racf_tmp"]
        .last()
        .reset_index()
        .rename(columns={"racf_tmp": "rolling_acf"})
    )
    merged = monthly[["month", col_flow]].merge(me_acf, on="month").dropna()
    if len(merged) > 5:
        corr = merged[col_flow].corr(merged["rolling_acf"])
        flow_acf_corrs[fund] = {
            "corr": corr,
            "n": len(merged),
            "flow_col": col_flow,
            "data": merged
        }
        print(f"  {fund}: fund_flow% vs rolling-ACF  r = {corr:.4f}  "
              f"(n={len(merged)} months)  "
              f"→ {'flow → inefficiency' if corr > 0.15 else 'flow → efficiency' if corr < -0.15 else 'weak link'}")

daily.drop(columns=["month","racf_tmp"], errors="ignore", inplace=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SUB-PERIOD EFFICIENCY (High Rate vs Low Rate Regime)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 4: Sub-period efficiency (rate regimes) ──────────────")

# Split into:
# High rate period: 2022-04-01 → 2023-12-31 (SBP rate 12%–22%)
# Low rate period : 2021-01-01 → 2022-03-31 and 2024-01-01 → 2025-10-01

high_rate_mask = (daily["date"] >= "2022-04-01") & (daily["date"] <= "2023-12-31")
low_rate_mask  = ~high_rate_mask

print("\n  [High-rate regime: Apr 2022 – Dec 2023]")
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r_high = daily.loc[high_rate_mask, col].dropna().values
    rt = runs_test(r_high, name=fund)
    vr = variance_ratio_test(r_high, q=2, name=fund)
    H  = hurst_exponent(r_high, name=fund)
    print(f"  {fund}: Runs Z={rt['Z']:>7.4f} (p={rt['p_value']:.3f}) | "
          f"VR={vr['VR']:.4f} (p={vr['p_value']:.3f}) | "
          f"H={H['H']:.4f}")

print("\n  [Low-rate regime: Jan 2021–Mar 2022 + Jan 2024–Oct 2025]")
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    r_low = daily.loc[low_rate_mask, col].dropna().values
    rt = runs_test(r_low, name=fund)
    vr = variance_ratio_test(r_low, q=2, name=fund)
    H  = hurst_exponent(r_low, name=fund)
    print(f"  {fund}: Runs Z={rt['Z']:>7.4f} (p={rt['p_value']:.3f}) | "
          f"VR={vr['VR']:.4f} (p={vr['p_value']:.3f}) | "
          f"H={H['H']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: Generating efficiency figures ──────────────────────")

# ── E1. ACF plot for all 3 funds ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Autocorrelation Function — Daily NAV Log Returns\n"
             "(bars outside dashed lines → significant autocorrelation → inefficiency)",
             fontweight="bold")
for i, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                   ("NBP","nav_return_nbp"),
                                   ("NIT","nav_return_nti")]):
    r  = daily[col].dropna().values
    n  = len(r)
    ci = 1.96 / np.sqrt(n)
    acf_vals = []
    for k in range(1, 21):
        acf_vals.append(np.corrcoef(r[k:], r[:-k])[0, 1])
    ax = axes[i]
    colors = ["#e74c3c" if abs(v) > ci else FUND_COLORS[fund] for v in acf_vals]
    ax.bar(range(1, 21), acf_vals, color=colors, alpha=0.8)
    ax.axhline( ci, color="black", linewidth=0.8, linestyle="--", label="95% CI")
    ax.axhline(-ci, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0,   color="black", linewidth=0.5)
    ax.set_title(f"{fund}\nLjung-Box p={acf_results[fund]['p_value']:.4f}  "
                 f"→ {acf_results[fund]['verdict'].split('(')[0].strip()}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("E1_acf_nav_returns.png")

# ── E2. Variance Ratio across q values ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
q_vals = [2, 4, 8, 16]
for fund in ["AKD", "NBP", "NIT"]:
    vr_vals = [vr["VR"] for vr in vr_results[fund]]
    ax.plot(q_vals, vr_vals, marker="o", color=FUND_COLORS[fund],
            linewidth=1.5, markersize=7, label=fund)
    # Mark significant ones
    sig = [vr["p_value"] < 0.05 for vr in vr_results[fund]]
    ax.scatter([q_vals[i] for i, s in enumerate(sig) if s],
               [vr_vals[i] for i, s in enumerate(sig) if s],
               color=FUND_COLORS[fund], s=120, marker="*", zorder=5)
ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="VR=1 (random walk)")
ax.fill_between(q_vals, 0.85, 1.15, alpha=0.08, color="gray", label="±15% band")
ax.set_xlabel("Holding period q (days)")
ax.set_ylabel("Variance Ratio VR(q)")
ax.set_title("Variance Ratio Test across Holding Periods\n"
             "(⭐ = statistically significant deviation from random walk)")
ax.legend()
ax.set_xticks(q_vals)
plt.tight_layout()
savefig("E2_variance_ratio.png")

# ── E3. Rolling autocorrelation over time ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    racf = rolling_autocorrelation(daily[col], daily["date"], window=60, lag=1)
    ax.plot(daily["date"], racf, color=FUND_COLORS[fund], linewidth=1, alpha=0.85, label=fund)
ax.axhline(0,     color="black", linewidth=0.7)
ax.axhline( 0.10, color="gray",  linewidth=0.8, linestyle="--", label="±0.10 band")
ax.axhline(-0.10, color="gray",  linewidth=0.8, linestyle="--")
# Shade high-rate period
ax.axvspan(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-12-31"),
           alpha=0.10, color="#e74c3c", label="High-rate regime")
ax.set_title("Rolling 60-Day Lag-1 Autocorrelation of NAV Returns\n"
             "(near zero → efficient; persistent deviation → inefficient)")
ax.set_ylabel("Autocorrelation (lag 1)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("E3_rolling_autocorrelation.png")

# ── E4. Fund flow vs rolling ACF scatter ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Monthly Fund Flow (% AUM) vs 60-Day Rolling Lag-1 Autocorrelation\n"
             "(tests whether fund flows affect market efficiency)",
             fontweight="bold")
for i, fund in enumerate(["AKD", "NBP", "NIT"]):
    ax = axes[i]
    if fund in flow_acf_corrs:
        d    = flow_acf_corrs[fund]["data"]
        fcol = flow_acf_corrs[fund]["flow_col"]
        corr = flow_acf_corrs[fund]["corr"]
        ax.scatter(d[fcol] * 100, d["rolling_acf"],
                   color=FUND_COLORS[fund], alpha=0.7, s=55)
        m, b = np.polyfit(d[fcol], d["rolling_acf"], 1)
        xl   = np.linspace(d[fcol].min(), d[fcol].max(), 100)
        ax.plot(xl * 100, m * xl + b, "k--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"{fund}  (r = {corr:.3f})")
        ax.set_xlabel("Fund flow (% of AUM)")
        ax.set_ylabel("Rolling ACF (lag 1)")
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(fund)
plt.tight_layout()
savefig("E4_flow_vs_acf_scatter.png")

# ── E5. Hurst exponent summary bar ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
hurst_bars = {fund: hurst_results[fund]["H"] for fund in ["AKD","NBP","NIT"]
              if hurst_results[fund]}
# Also add stocks
stock_hurst = {}
for sym in returns_wide.columns[:12]:  # top 12 for readability
    r = returns_wide[sym].dropna().values
    H = hurst_exponent(r, name=sym)
    if H:
        stock_hurst[sym] = H["H"]

all_bars = list(hurst_bars.items()) + list(stock_hurst.items())
labels   = [x[0] for x in all_bars]
hvals    = [x[1] for x in all_bars]
bar_cols = (
    ["#1f77b4" if l in ["AKD","NBP","NIT"] else
     "#e74c3c" if v > 0.55 else
     "#2ecc71" if v < 0.45 else
     "#95a5a6"
     for l, v in all_bars]
)
bars = ax.bar(labels, hvals, color=bar_cols, alpha=0.85, edgecolor="white")
ax.axhline(0.5, color="black", linewidth=1.5, linestyle="--", label="H=0.5 (random walk)")
ax.axhspan(0.45, 0.55, alpha=0.08, color="gray", label="Random walk band ±0.05")
ax.set_ylabel("Hurst Exponent H")
ax.set_title("Hurst Exponents — Funds (blue) and KSE-30 Stocks\n"
             "Red = persistent (H>0.55), Green = mean-reverting (H<0.45), Gray = RW")
ax.legend(fontsize=8)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
for bar, val in zip(bars, hvals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
savefig("E5_hurst_exponents.png")

# ── E6. Stock-level efficiency summary heatmap ────────────────────────────────
verdict_map = {"Efficient": 0, "Inefficient": 1}
heat_data = eff_df[["Runs verdict","VR verdict","LB verdict"]].copy()
heat_num  = heat_data.replace(verdict_map)

from matplotlib.colors import ListedColormap
cmap_bi = ListedColormap(["#2ecc71", "#e74c3c"])
fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(heat_num.T.astype(float), annot=False,
            cmap=cmap_bi, vmin=0, vmax=1,
            linewidths=0.5, ax=ax, cbar=False)
# Annotate manually
for ri, row_label in enumerate(heat_data.T.index):
    for ci, col_label in enumerate(heat_data.T.columns):
        val = heat_data.T.iloc[ri, ci]
        ax.text(ci + 0.5, ri + 0.5, val,
                ha="center", va="center", fontsize=7,
                color="white" if heat_num.T.iloc[ri, ci] == 1 else "black")
ax.set_title("Stock-Level Weak-Form Efficiency Test Results\n"
             "(Green = Efficient, Red = Inefficient at 5% level)")
ax.set_xlabel("Stock")
ax.set_ylabel("Test")
ax.tick_params(axis="x", labelsize=8, rotation=45)
ax.tick_params(axis="y", labelsize=9, rotation=0)
plt.tight_layout()
savefig("E6_stock_efficiency_heatmap.png")

# ── E7. Sub-period efficiency comparison ─────────────────────────────────────
print("\n[Table] Sub-period VR(2) values:")
print(f"  {'Fund':<6} {'Full VR':>10} {'High-rate VR':>14} {'Low-rate VR':>13}")
for fund, col in [("AKD","nav_return_akd"),
                  ("NBP","nav_return_nbp"),
                  ("NIT","nav_return_nti")]:
    full_vr = vr_results[fund][0]["VR"]
    r_hi = daily.loc[high_rate_mask, col].dropna().values
    r_lo = daily.loc[low_rate_mask,  col].dropna().values
    vr_hi = variance_ratio_test(r_hi, q=2)
    vr_lo = variance_ratio_test(r_lo, q=2)
    print(f"  {fund:<6} {full_vr:>10.4f} {vr_hi['VR']:>14.4f} {vr_lo['VR']:>13.4f}")


# ── E8. Summary efficiency table ─────────────────────────────────────────────
print("\n[Table] Fund-Level Efficiency Summary (Full Period)")
print(f"\n{'Test':<22} {'AKD':>20} {'NBP':>20} {'NIT':>20}")
print("-" * 85)
for fund in ["AKD","NBP","NIT"]:
    pass  # collected below

rows = [
    ("Runs Z-stat",       {f: f"{runs_results[f]['Z']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("Runs p-value",      {f: f"{runs_results[f]['p_value']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("Runs verdict",      {f: runs_results[f]['verdict'].split('(')[0].strip() for f in ["AKD","NBP","NIT"]}),
    ("VR(2)",             {f: f"{vr_results[f][0]['VR']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("VR(4)",             {f: f"{vr_results[f][1]['VR']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("VR(2) p-value",     {f: f"{vr_results[f][0]['p_value']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("Ljung-Box Q p",     {f: f"{acf_results[f]['p_value']:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("ACF lag-1",         {f: f"{acf_results[f]['acf'][0]:.4f}" for f in ["AKD","NBP","NIT"]}),
    ("Hurst H",           {f: f"{hurst_results[f]['H']:.4f}" if hurst_results[f] else "N/A"
                           for f in ["AKD","NBP","NIT"]}),
    ("Hurst interpretation", {f: hurst_results[f]['interpretation'] if hurst_results[f] else "N/A"
                               for f in ["AKD","NBP","NIT"]}),
]
for label, vals in rows:
    print(f"{label:<22} {vals['AKD']:>20} {vals['NBP']:>20} {vals['NIT']:>20}")

print(f"\nAll figures saved to: {FIG_DIR}")
print("MARKET EFFICIENCY ANALYSIS COMPLETE.")
