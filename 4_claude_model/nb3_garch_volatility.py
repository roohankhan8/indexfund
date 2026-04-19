"""
NOTEBOOK 3 — GARCH Volatility Modelling
=========================================
Input  : daily_master.csv
Outputs: figures → ./figures/garch/

Models
  1. ARCH(1)          — baseline, only ARCH term
  2. GARCH(1,1)       — standard, most cited for PSX literature
  3. EGARCH(1,1)      — captures asymmetry (leverage effect)
  4. GARCH-M(1,1)     — risk-in-mean: volatility enters return equation

Applied to: daily NAV log returns of AKD, NBP, NIT
            and daily oil / USD/PKR log returns (as covariates)

Diagnostics
  - Ljung-Box Q test on standardised residuals
  - Engle ARCH-LM test
  - VaR backtesting (5% level)

Run: python nb3_garch_volatility.py
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
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "garch")
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
daily = pd.read_csv("processed_data/daily_master.csv", parse_dates=["date"])
daily = daily.sort_values("date").reset_index(drop=True)
print(f"Daily rows: {len(daily)}  {daily['date'].min().date()} → {daily['date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════
# GARCH FAMILY — NUMPY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def garch11_loglik(params, returns):
    """
    GARCH(1,1) negative log-likelihood (Gaussian innovations).
    params = [omega, alpha, beta]
    Constraint: omega>0, alpha>=0, beta>=0, alpha+beta<1
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    if np.any(sigma2 <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
    return -ll  # negative LL for minimisation


def arch1_loglik(params, returns):
    """ARCH(1): sigma2[t] = omega + alpha * r[t-1]^2"""
    omega, alpha = params
    if omega <= 0 or alpha < 0 or alpha >= 1:
        return 1e10
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2
    if np.any(sigma2 <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + returns**2/sigma2)
    return -ll


def egarch11_loglik(params, returns):
    """
    EGARCH(1,1) — Nelson (1991).
    log(sigma2[t]) = omega + alpha*(|z[t-1]| - E[|z|]) + gamma*z[t-1] + beta*log(sigma2[t-1])
    where z = r/sigma (standardised residual), E[|z|] = sqrt(2/pi)
    """
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1:
        return 1e10
    n = len(returns)
    log_sigma2 = np.zeros(n)
    log_sigma2[0] = np.log(np.var(returns))
    EZ = np.sqrt(2 / np.pi)
    for t in range(1, n):
        sigma_prev = np.exp(0.5 * log_sigma2[t-1])
        z_prev = returns[t-1] / (sigma_prev + 1e-10)
        log_sigma2[t] = (omega +
                         alpha * (abs(z_prev) - EZ) +
                         gamma * z_prev +
                         beta  * log_sigma2[t-1])
    sigma2 = np.exp(log_sigma2)
    if np.any(sigma2 <= 0) or np.any(np.isinf(sigma2)):
        return 1e10
    ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + returns**2/sigma2)
    return -ll


def garch_m_loglik(params, returns):
    """
    GARCH-M(1,1): r[t] = mu + lambda*sigma[t] + eps[t], eps ~ N(0, sigma2[t])
    sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1]
    params = [mu, lambda, omega, alpha, beta]
    """
    mu, lam, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    sigma2  = np.zeros(n)
    sigma2[0] = np.var(returns)
    eps      = np.zeros(n)
    for t in range(1, n):
        eps[t-1]  = returns[t-1] - mu - lam * np.sqrt(sigma2[t-1])
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    if np.any(sigma2 <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + eps**2/sigma2)
    return -ll


def fit_garch(returns, model="garch11"):
    """Fit model to returns series via numerical MLE."""
    var0  = np.var(returns)
    mu0   = np.mean(returns)

    if model == "arch1":
        x0     = [var0 * 0.5, 0.2]
        bounds = [(1e-8, None), (1e-6, 0.999)]
        res    = minimize(arch1_loglik, x0, args=(returns,),
                          method="L-BFGS-B", bounds=bounds)
        omega, alpha = res.x
        n = len(returns)
        sigma2 = np.zeros(n); sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2
        return {"model": "ARCH(1)", "params": {"omega": omega, "alpha": alpha},
                "sigma2": sigma2, "aic": 2*2 + 2*res.fun,
                "bic": 2*np.log(n) + 2*res.fun, "nll": res.fun}

    elif model == "garch11":
        x0     = [var0*0.1, 0.1, 0.8]
        bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
        res    = minimize(garch11_loglik, x0, args=(returns,),
                          method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = res.x
        n = len(returns)
        sigma2 = np.zeros(n); sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = omega + alpha*returns[t-1]**2 + beta*sigma2[t-1]
        persistence = alpha + beta
        return {"model": "GARCH(1,1)", "params": {"omega":omega,"alpha":alpha,"beta":beta},
                "sigma2": sigma2, "persistence": persistence,
                "aic": 2*3 + 2*res.fun, "bic": 3*np.log(n) + 2*res.fun, "nll": res.fun}

    elif model == "egarch11":
        x0     = [np.log(var0)*0.1, 0.1, -0.05, 0.9]
        bounds = [(-10,10),(-2,2),(-2,2),(-0.999,0.999)]
        res    = minimize(egarch11_loglik, x0, args=(returns,),
                          method="L-BFGS-B", bounds=bounds)
        omega, alpha, gamma, beta = res.x
        n = len(returns)
        log_s2 = np.zeros(n); log_s2[0] = np.log(var0)
        EZ = np.sqrt(2/np.pi)
        for t in range(1, n):
            s_prev = np.exp(0.5*log_s2[t-1])
            z      = returns[t-1]/(s_prev+1e-10)
            log_s2[t] = omega + alpha*(abs(z)-EZ) + gamma*z + beta*log_s2[t-1]
        sigma2 = np.exp(log_s2)
        lev_effect = "YES" if gamma < 0 else "NO"
        return {"model": "EGARCH(1,1)",
                "params": {"omega":omega,"alpha":alpha,"gamma":gamma,"beta":beta},
                "sigma2": sigma2, "leverage_effect": lev_effect,
                "aic": 2*4 + 2*res.fun, "bic": 4*np.log(n)+2*res.fun, "nll": res.fun}

    elif model == "garch_m":
        x0     = [mu0, 0.0, var0*0.1, 0.1, 0.8]
        bounds = [(None,None),(None,None),(1e-8,None),(1e-6,0.999),(1e-6,0.999)]
        res    = minimize(garch_m_loglik, x0, args=(returns,),
                          method="L-BFGS-B", bounds=bounds)
        mu_p, lam, omega, alpha, beta = res.x
        n = len(returns)
        sigma2 = np.zeros(n); sigma2[0] = var0; eps = np.zeros(n)
        for t in range(1,n):
            eps[t-1]  = returns[t-1] - mu_p - lam*np.sqrt(sigma2[t-1])
            sigma2[t] = omega + alpha*eps[t-1]**2 + beta*sigma2[t-1]
        return {"model": "GARCH-M(1,1)",
                "params": {"mu":mu_p,"lambda":lam,"omega":omega,"alpha":alpha,"beta":beta},
                "sigma2": sigma2, "risk_premium_lambda": lam,
                "aic": 2*5+2*res.fun, "bic": 5*np.log(n)+2*res.fun, "nll": res.fun}


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def ljung_box(resid_std, lags=10):
    """Ljung-Box Q statistic on standardised residuals."""
    n = len(resid_std)
    acf_vals = []
    for k in range(1, lags + 1):
        r = np.corrcoef(resid_std[k:], resid_std[:-k])[0, 1]
        acf_vals.append(r)
    Q = n * (n + 2) * sum(rk**2 / (n - k)
                          for k, rk in enumerate(acf_vals, 1))
    p_val = 1 - stats.chi2.cdf(Q, df=lags)
    return Q, p_val

def arch_lm_test(resid, lags=5):
    """Engle's ARCH-LM test: H0 = no ARCH effects."""
    sq = resid ** 2
    n  = len(sq)
    y  = sq[lags:]
    X  = np.column_stack([sq[lags - i - 1 : n - i - 1] for i in range(lags)])
    X  = np.column_stack([np.ones(len(y)), X])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2  = 1 - ss_res / ss_tot
    LM  = n * r2
    pv  = 1 - stats.chi2.cdf(LM, df=lags)
    return LM, pv

def var_backtest(returns, sigma2, alpha=0.05):
    """
    Historical VaR at alpha level using GARCH volatility.
    Count exceedances and compare to expected.
    """
    var_forecast = stats.norm.ppf(alpha) * np.sqrt(sigma2)
    exceedances  = np.sum(returns < var_forecast)
    expected     = alpha * len(returns)
    exceed_rate  = exceedances / len(returns)
    return {"VaR_exceedances": exceedances,
            "expected": expected,
            "exceed_rate": exceed_rate,
            "kupiec_pass": abs(exceed_rate - alpha) < 0.02}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — FIT ALL MODELS TO ALL SERIES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: Fitting GARCH family models ────────────────────────")
series_map = {
    "AKD": "nav_return_akd",
    "NBP": "nav_return_nbp",
    "NIT": "nav_return_nti",
}
all_fits = {}

for series_name, col in series_map.items():
    print(f"\n  {series_name} ({col})")
    r = daily[col].dropna().values * 100   # convert to percent for numerical stability
    dates_r = daily.loc[daily[col].notna(), "date"].values

    fits = {}
    for mname in ["arch1", "garch11", "egarch11", "garch_m"]:
        print(f"    Fitting {mname} …", end=" ", flush=True)
        fit = fit_garch(r, model=mname)
        # Diagnostics
        sigma = np.sqrt(np.maximum(fit["sigma2"], 1e-12))
        std_resid = r / sigma
        lb_q, lb_p = ljung_box(std_resid, lags=10)
        lm_stat, lm_p = arch_lm_test(std_resid, lags=5)
        var_bt = var_backtest(r, fit["sigma2"], alpha=0.05)
        fit["lb_q"] = lb_q; fit["lb_p"] = lb_p
        fit["lm_stat"] = lm_stat; fit["lm_p"] = lm_p
        fit["var_bt"] = var_bt
        fit["std_resid"] = std_resid
        fit["dates"] = dates_r
        fit["returns"] = r
        fits[mname] = fit
        print(f"AIC={fit['aic']:.2f}  BIC={fit['bic']:.2f}")

    all_fits[series_name] = fits

    # Summary table
    print(f"\n  {'Model':16s} {'AIC':>10} {'BIC':>10} {'LB p':>8} {'ARCH-LM p':>10} {'VaR exc%':>9} {'Best?'}")
    print(f"  {'-'*65}")
    aic_vals = {k: v["aic"] for k, v in fits.items()}
    best_model = min(aic_vals, key=aic_vals.get)
    for mname, fit in fits.items():
        flag = "<-- best AIC" if mname == best_model else ""
        print(f"  {fit['model']:16s} {fit['aic']:>10.2f} {fit['bic']:>10.2f} "
              f"{fit['lb_p']:>8.4f} {fit['lm_p']:>10.4f} "
              f"{fit['var_bt']['exceed_rate']*100:>9.2f}%  {flag}")

    # Extra info for best model
    best = fits[best_model]
    print(f"\n  Best model params ({best['model']}):")
    for k, v in best["params"].items():
        print(f"    {k} = {v:.6f}")
    if "persistence" in best:
        print(f"    persistence (α+β) = {best['persistence']:.6f}"
              f"  {'(near unit root — high persistence)' if best['persistence'] > 0.95 else ''}")
    if "leverage_effect" in best:
        print(f"    leverage effect: {best['leverage_effect']}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Generating GARCH figures ──────────────────────────")

# ── G1. Returns + conditional volatility (GARCH 1,1) ────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Daily NAV Returns & GARCH(1,1) Conditional Volatility", fontweight="bold")
for row, (series_name, col) in enumerate(series_map.items()):
    r  = daily[col].dropna().values * 100
    dt = daily.loc[daily[col].notna(), "date"].values
    fit = all_fits[series_name]["garch11"]
    vol = np.sqrt(fit["sigma2"])

    ax_r = axes[row][0]
    ax_r.fill_between(dt, r, alpha=0.5, color=FUND_COLORS[series_name])
    ax_r.axhline(0, color="black", linewidth=0.7)
    ax_r.set_title(f"{series_name} — Daily NAV Log Returns (%)")
    ax_r.set_ylabel("Return (%)")
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax_r.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    ax_v = axes[row][1]
    ax_v.plot(dt, vol, color=FUND_COLORS[series_name], linewidth=1)
    ax_v.set_title(f"{series_name} — GARCH(1,1) Conditional Std Dev (%)")
    ax_v.set_ylabel("Cond. Std Dev (%)")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax_v.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("G1_returns_and_garch_vol.png")

# ── G2. EGARCH volatility + leverage effect test ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("EGARCH(1,1) — News Impact Curve (Leverage Effect)", fontweight="bold")
for i, series_name in enumerate(["AKD", "NBP", "NIT"]):
    fit  = all_fits[series_name]["egarch11"]
    p    = fit["params"]
    EZ   = np.sqrt(2/np.pi)
    # News impact curve: conditional on sigma2_prev = unconditional variance
    z_range = np.linspace(-4, 4, 200)
    log_var_base = np.mean(np.log(fit["sigma2"]))
    impact = np.exp(log_var_base + p["alpha"]*(np.abs(z_range)-EZ) + p["gamma"]*z_range)
    ax = axes[i]
    ax.plot(z_range, impact, color=FUND_COLORS[series_name], linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title(f"{series_name}\nγ={p['gamma']:.4f} "
                 f"({'leverage' if p['gamma']<0 else 'no leverage'})")
    ax.set_xlabel("Shock (z)")
    ax.set_ylabel("Cond. variance impact")
plt.tight_layout()
savefig("G2_egarch_news_impact.png")

# ── G3. Standardised residual diagnostics ────────────────────────────────────
for series_name in ["AKD", "NBP", "NIT"]:
    fit = all_fits[series_name]["garch11"]
    std_r = fit["std_resid"]
    dt    = fit["dates"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{series_name} — GARCH(1,1) Standardised Residual Diagnostics",
                 fontweight="bold")

    # Time series of std residuals
    axes[0][0].plot(dt, std_r, color=FUND_COLORS[series_name], linewidth=0.7, alpha=0.8)
    axes[0][0].axhline(0, color="black", linewidth=0.7)
    axes[0][0].axhline( 1.96, color="red", linewidth=0.8, linestyle="--")
    axes[0][0].axhline(-1.96, color="red", linewidth=0.8, linestyle="--")
    axes[0][0].set_title("Standardised Residuals over Time")
    axes[0][0].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axes[0][0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Histogram + normal
    axes[0][1].hist(std_r, bins=50, density=True, color=FUND_COLORS[series_name],
                    alpha=0.7, edgecolor="white")
    x = np.linspace(std_r.min(), std_r.max(), 200)
    axes[0][1].plot(x, stats.norm.pdf(x), "k--", linewidth=1.5, label="N(0,1)")
    axes[0][1].set_title("Distribution of Std Residuals")
    axes[0][1].legend()

    # QQ plot
    (osm, osr), (slope, intercept, r) = stats.probplot(std_r, dist="norm")
    axes[1][0].scatter(osm, osr, s=8, alpha=0.5, color=FUND_COLORS[series_name])
    axes[1][0].plot(osm, slope*np.array(osm)+intercept, "k--", linewidth=1)
    axes[1][0].set_title(f"Q-Q Plot  (R²={r**2:.4f})")
    axes[1][0].set_xlabel("Theoretical quantiles")
    axes[1][0].set_ylabel("Sample quantiles")

    # ACF of squared std residuals
    max_lag = 20
    acf_sq = [np.corrcoef(std_r[k:]**2, std_r[:-k]**2)[0,1]
              for k in range(1, max_lag+1)]
    ci = 1.96 / np.sqrt(len(std_r))
    axes[1][1].bar(range(1, max_lag+1), acf_sq,
                   color=FUND_COLORS[series_name], alpha=0.8)
    axes[1][1].axhline( ci, color="red", linewidth=0.8, linestyle="--")
    axes[1][1].axhline(-ci, color="red", linewidth=0.8, linestyle="--")
    axes[1][1].axhline(0, color="black", linewidth=0.5)
    axes[1][1].set_title("ACF of Squared Std Residuals\n(should be ~0 if GARCH captured vol)")
    axes[1][1].set_xlabel("Lag")

    plt.tight_layout()
    savefig(f"G3_{series_name}_residual_diagnostics.png")

# ── G4. VaR backtest visualisation ───────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle("GARCH(1,1) 5% Value-at-Risk Backtest", fontweight="bold")
for i, (series_name, col) in enumerate(series_map.items()):
    r   = daily[col].dropna().values * 100
    dt  = daily.loc[daily[col].notna(), "date"].values
    fit = all_fits[series_name]["garch11"]
    var_line = stats.norm.ppf(0.05) * np.sqrt(fit["sigma2"])
    exceed   = r < var_line

    ax = axes[i]
    ax.fill_between(dt, r, 0, where=(r >= 0), alpha=0.4,
                    color=FUND_COLORS[series_name], label="Return")
    ax.fill_between(dt, r, 0, where=(r < 0), alpha=0.4, color="#e74c3c")
    ax.plot(dt, var_line, color="black", linewidth=1.2, label="VaR 5%")
    ax.scatter(dt[exceed], r[exceed], color="black", zorder=5, s=10,
               label=f"Exceedances ({exceed.sum()})")
    ax.set_title(f"{series_name} — "
                 f"Exceedance rate={exceed.mean()*100:.1f}% (expected 5%)")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("G4_var_backtest.png")

# ── G5. Model comparison AIC/BIC heatmap ─────────────────────────────────────
model_labels = ["ARCH(1)", "GARCH(1,1)", "EGARCH(1,1)", "GARCH-M(1,1)"]
model_keys   = ["arch1",   "garch11",    "egarch11",     "garch_m"]
aic_matrix   = np.array([[all_fits[s][m]["aic"] for m in model_keys]
                          for s in ["AKD","NBP","NIT"]])
bic_matrix   = np.array([[all_fits[s][m]["bic"] for m in model_keys]
                          for s in ["AKD","NBP","NIT"]])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Model Selection — AIC and BIC (lower is better)", fontweight="bold")
for ax, matrix, title in [(axes[0], aic_matrix, "AIC"),
                            (axes[1], bic_matrix, "BIC")]:
    # Normalise row-wise so the best model per series is obvious
    row_min = matrix.min(axis=1, keepdims=True)
    norm    = matrix - row_min
    sns.heatmap(norm, annot=matrix.round(1), fmt=".1f",
                xticklabels=model_labels, yticklabels=["AKD","NBP","NIT"],
                cmap="YlOrRd", ax=ax, cbar_kws={"label": "Δ from best"})
    ax.set_title(f"{title} — raw values annotated, colour = Δ from best per row")
plt.tight_layout()
savefig("G5_model_selection_aic_bic.png")

# ── G6. Volatility comparison across models (AKD) ────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
col  = "nav_return_akd"
r    = daily[col].dropna().values * 100
dt   = daily.loc[daily[col].notna(), "date"].values
linestyles = ["-", "--", "-.", ":"]
colors_m   = ["#2ecc71","#3498db","#e74c3c","#f39c12"]
for (mkey, mlabel), ls, col_m in zip(
    [("arch1","ARCH(1)"),("garch11","GARCH(1,1)"),
     ("egarch11","EGARCH(1,1)"),("garch_m","GARCH-M(1,1)")],
    linestyles, colors_m
):
    vol = np.sqrt(all_fits["AKD"][mkey]["sigma2"])
    ax.plot(dt, vol, linewidth=1.2, linestyle=ls, color=col_m, label=mlabel)
ax.set_title("AKD — Conditional Volatility: All GARCH Models Compared")
ax.set_ylabel("Cond. Std Dev (% return)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
savefig("G6_vol_model_comparison_AKD.png")

# ── G7. Macro correlation with fund volatility ────────────────────────────────
print("\n[Summary] Correlation: GARCH(1,1) conditional vol vs macro")
vol_akd = np.sqrt(all_fits["AKD"]["garch11"]["sigma2"])
vol_nbp = np.sqrt(all_fits["NBP"]["garch11"]["sigma2"])
vol_nti = np.sqrt(all_fits["NIT"]["garch11"]["sigma2"])

df_vol = daily[["date","oil_log_return","usdpkr_log_return",
                 "nav_return_akd","nav_return_nbp","nav_return_nti"]].dropna().copy()
df_vol["vol_akd"] = vol_akd[:len(df_vol)]
df_vol["vol_nbp"] = vol_nbp[:len(df_vol)]
df_vol["vol_nti"] = vol_nti[:len(df_vol)]

for fund_vol in ["vol_akd","vol_nbp","vol_nti"]:
    for macro in ["oil_log_return","usdpkr_log_return"]:
        r_val = np.corrcoef(df_vol[fund_vol], df_vol[macro])[0,1]
        print(f"  {fund_vol} vs {macro}: r = {r_val:.4f}")

print(f"\nAll figures saved to: {FIG_DIR}")
print("GARCH VOLATILITY MODELLING COMPLETE.")
