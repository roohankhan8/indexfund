"""
NOTEBOOK 4 — Stock-Level Correlation Analysis & Markowitz Optimisation
=======================================================================
Input  : kse30_stocks_daily.csv
Outputs: figures → ./figures/portfolio/
         weights → portfolio_weights.csv

Sections
  1. Sector-level return analysis
  2. Stock correlation heatmap (pivot to wide-form returns)
  3. Return distribution comparison across stocks
  4. Markowitz efficient frontier (numpy mean-variance optimisation)
  5. Three portfolio comparison:
       A. Equal-weight
       B. Market-cap weight (using average IDX WT %)
       C. Markowitz minimum-variance
       D. Markowitz maximum-Sharpe
  6. Rolling Sharpe ratio over time for each portfolio
  7. Portfolio weights table → saved to CSV

Run: python nb4_portfolio_optimisation.py
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
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "portfolio")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 8})

RISK_FREE = 0.105 / 252   # SBP rate ~10.5% annualised → daily

# ── load ─────────────────────────────────────────────────────────────────────
stocks = pd.read_csv("processed_data/kse30_stocks_daily.csv", parse_dates=["date"])
stocks = stocks.sort_values(["symbol","date"]).reset_index(drop=True)
print(f"Stock rows: {len(stocks):,}  Symbols: {stocks['symbol'].nunique()}")
print(f"Date range: {stocks['date'].min().date()} → {stocks['date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — SELECT ACTIVE CONSTITUENTS
# ═══════════════════════════════════════════════════════════════════════════

# Keep only stocks that appear in the index for at least 80% of trading days
# (removes stocks that briefly rotated in/out)
all_dates = stocks["date"].drop_duplicates()
n_dates   = len(all_dates)
count_per_sym = stocks.groupby("symbol")["date"].count()
active_syms   = count_per_sym[count_per_sym >= 0.80 * n_dates].index.tolist()
print(f"\nActive symbols (≥80% of trading days): {len(active_syms)}")
print(active_syms)

stocks = stocks[stocks["symbol"].isin(active_syms)].copy()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — PIVOT TO WIDE-FORM RETURNS
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: Building returns matrix ───────────────────────────")
returns_wide = (
    stocks.pivot_table(index="date", columns="symbol", values="log_return")
    .sort_index()
)
# Drop dates with >30% missing
returns_wide = returns_wide.dropna(thresh=int(0.70 * returns_wide.shape[1]))
# Forward-fill remaining gaps (trading halts, etc.) then drop any remaining
returns_wide = returns_wide.ffill(limit=2).dropna(axis=1, thresh=int(0.90*len(returns_wide)))
print(f"Returns matrix: {returns_wide.shape[0]} dates × {returns_wide.shape[1]} stocks")
print(f"Remaining NaN: {returns_wide.isnull().sum().sum()}")

# Fill residual NaN with 0 (stock suspended = 0 return day)
returns_wide = returns_wide.fillna(0)

SYMBOLS = returns_wide.columns.tolist()
N = len(SYMBOLS)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE STATS PER STOCK
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Per-stock descriptive stats ────────────────────────")
desc = pd.DataFrame(index=SYMBOLS)
for sym in SYMBOLS:
    r = returns_wide[sym]
    desc.loc[sym, "mean_daily"]     = r.mean()
    desc.loc[sym, "std_daily"]      = r.std()
    desc.loc[sym, "mean_annual"]    = r.mean() * 252
    desc.loc[sym, "vol_annual"]     = r.std() * np.sqrt(252)
    desc.loc[sym, "sharpe_annual"]  = ((r.mean() - RISK_FREE) * 252) / (r.std() * np.sqrt(252))
    desc.loc[sym, "skewness"]       = r.skew()
    desc.loc[sym, "kurtosis"]       = r.kurtosis()
    desc.loc[sym, "max_drawdown"]   = (r.cumsum() - r.cumsum().cummax()).min()

desc = desc.astype(float)
print("\n[Table] Top 10 by Sharpe ratio:")
print(desc.sort_values("sharpe_annual", ascending=False).head(10)[
    ["mean_annual","vol_annual","sharpe_annual","skewness"]].round(4).to_string())
print("\n[Table] Bottom 10 by Sharpe ratio:")
print(desc.sort_values("sharpe_annual").head(10)[
    ["mean_annual","vol_annual","sharpe_annual","skewness"]].round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 3: Correlation analysis ──────────────────────────────")
corr_matrix = returns_wide.corr()

# ── P1. Full correlation heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="RdYlGn", center=0,
            vmin=-0.5, vmax=1, annot=False, linewidths=0.2,
            ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("KSE-30 Constituent Stock Pairwise Correlation (Daily Log Returns)")
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.tick_params(axis="y", labelsize=7, rotation=0)
plt.tight_layout()
savefig("P1_stock_correlation_heatmap.png")

# Correlation statistics
corr_vals = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
print(f"\nCorrelation stats across {len(corr_vals)} pairs:")
print(f"  Mean: {corr_vals.mean():.4f}  Std: {corr_vals.std():.4f}")
print(f"  Min: {corr_vals.min():.4f}  Max: {corr_vals.max():.4f}")
print(f"  % positive pairs: {(corr_vals>0).mean()*100:.1f}%")
print(f"  % pairs with r > 0.5: {(corr_vals>0.5).mean()*100:.1f}%")

# ── P2. Clustermap (hierarchical clustering) ─────────────────────────────────
fig = sns.clustermap(corr_matrix, cmap="RdYlGn", center=0,
                     vmin=-0.5, vmax=1, figsize=(16, 16),
                     dendrogram_ratio=0.1, annot=False, linewidths=0.2,
                     cbar_pos=(0.02, 0.85, 0.02, 0.1))
fig.ax_heatmap.set_title("Hierarchically Clustered Correlation Matrix", pad=60)
fig.ax_heatmap.tick_params(axis="x", labelsize=7)
fig.ax_heatmap.tick_params(axis="y", labelsize=7)
fig.savefig(os.path.join(FIG_DIR, "P2_clustered_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved → {os.path.join(FIG_DIR, 'P2_clustered_correlation.png')}")

# ── P3. Return-risk scatter ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(desc["vol_annual"] * 100, desc["mean_annual"] * 100,
                c=desc["sharpe_annual"], cmap="RdYlGn",
                s=80, alpha=0.85, edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Sharpe ratio")
for sym in SYMBOLS:
    ax.annotate(sym, (desc.loc[sym,"vol_annual"]*100, desc.loc[sym,"mean_annual"]*100),
                fontsize=6, ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points")
# Risk-free line
ax.axhline(RISK_FREE * 252 * 100, color="gray", linewidth=0.8,
           linestyle="--", label=f"Risk-free (~{RISK_FREE*252*100:.1f}%)")
ax.set_xlabel("Annualised Volatility (%)")
ax.set_ylabel("Annualised Mean Return (%)")
ax.set_title("Return–Risk Scatter — KSE-30 Constituents")
ax.legend()
plt.tight_layout()
savefig("P3_return_risk_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MARKOWITZ EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 4: Markowitz optimisation ────────────────────────────")

mu    = returns_wide.mean().values          # daily expected returns
Sigma = returns_wide.cov().values           # daily covariance matrix

def portfolio_stats(w):
    ret = w @ mu * 252
    vol = np.sqrt(w @ Sigma @ w) * np.sqrt(252)
    sr  = (ret - RISK_FREE * 252) / vol
    return ret, vol, sr

def neg_sharpe(w):
    _, vol, sr = portfolio_stats(w)
    return -sr

def port_vol(w):
    return np.sqrt(w @ Sigma @ w) * np.sqrt(252)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds      = tuple((0.0, 0.20) for _ in range(N))   # max 20% in any stock
w0          = np.ones(N) / N

# ── Minimum Variance ──────────────────────────────────────────────────────────
res_mv = minimize(port_vol, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000, "ftol": 1e-9})
w_mv  = res_mv.x
r_mv, v_mv, sr_mv = portfolio_stats(w_mv)
print(f"\nMinimum Variance Portfolio:")
print(f"  Return={r_mv*100:.2f}%  Vol={v_mv*100:.2f}%  Sharpe={sr_mv:.4f}")

# ── Maximum Sharpe ────────────────────────────────────────────────────────────
res_ms = minimize(neg_sharpe, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000, "ftol": 1e-9})
w_ms  = res_ms.x
r_ms, v_ms, sr_ms = portfolio_stats(w_ms)
print(f"\nMaximum Sharpe Portfolio:")
print(f"  Return={r_ms*100:.2f}%  Vol={v_ms*100:.2f}%  Sharpe={sr_ms:.4f}")

# ── Equal weight ──────────────────────────────────────────────────────────────
w_eq = np.ones(N) / N
r_eq, v_eq, sr_eq = portfolio_stats(w_eq)
print(f"\nEqual Weight Portfolio:")
print(f"  Return={r_eq*100:.2f}%  Vol={v_eq*100:.2f}%  Sharpe={sr_eq:.4f}")

# ── Market-cap weight (average IDX WT % per stock) ────────────────────────────
avg_wt = (
    stocks[stocks["symbol"].isin(SYMBOLS)]
    .groupby("symbol")["weight_pct"]
    .mean()
    .reindex(SYMBOLS)
    .fillna(0)
)
avg_wt = avg_wt.values
avg_wt = avg_wt / avg_wt.sum()
r_mc, v_mc, sr_mc = portfolio_stats(avg_wt)
print(f"\nMarket-Cap Weight Portfolio:")
print(f"  Return={r_mc*100:.2f}%  Vol={v_mc*100:.2f}%  Sharpe={sr_mc:.4f}")

# ── Efficient frontier (Monte Carlo simulation) ───────────────────────────────
print("\nGenerating efficient frontier (10,000 random portfolios) …")
np.random.seed(42)
n_sim = 10_000
sim_ret = np.zeros(n_sim)
sim_vol = np.zeros(n_sim)
sim_sr  = np.zeros(n_sim)
for i in range(n_sim):
    w = np.random.dirichlet(np.ones(N))
    r, v, sr = portfolio_stats(w)
    sim_ret[i] = r; sim_vol[i] = v; sim_sr[i] = sr

# Frontier portfolios (target return sweep)
target_returns = np.linspace(sim_ret.min(), sim_ret.max(), 60)
frontier_vols  = []
for target in target_returns:
    con = constraints + [{"type":"eq","fun":lambda w,t=target: portfolio_stats(w)[0]-t}]
    res = minimize(port_vol, w0, method="SLSQP",
                   bounds=bounds, constraints=con,
                   options={"maxiter":500, "ftol":1e-8})
    if res.success:
        frontier_vols.append(port_vol(res.x))
    else:
        frontier_vols.append(np.nan)
frontier_vols = np.array(frontier_vols)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PORTFOLIO FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: Portfolio figures ──────────────────────────────────")

# ── P4. Efficient frontier ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(sim_vol * 100, sim_ret * 100, c=sim_sr,
                cmap="RdYlGn", alpha=0.2, s=4)
plt.colorbar(sc, ax=ax, label="Sharpe ratio")
valid = ~np.isnan(frontier_vols)
ax.plot(frontier_vols[valid] * 100, target_returns[valid] * 100,
        "b-", linewidth=2.5, label="Efficient frontier", zorder=5)
# Plot the 4 portfolios
port_pts = [
    (v_mv,  r_mv,  sr_mv,  "Min Variance",  "^", "#3498db"),
    (v_ms,  r_ms,  sr_ms,  "Max Sharpe",    "*", "#e74c3c"),
    (v_eq,  r_eq,  sr_eq,  "Equal Weight",  "s", "#2ecc71"),
    (v_mc,  r_mc,  sr_mc,  "Market Cap",    "D", "#f39c12"),
]
for vol_p, ret_p, sr_p, label, marker, color in port_pts:
    ax.scatter(vol_p * 100, ret_p * 100, marker=marker, s=180,
               color=color, zorder=10, label=f"{label} (SR={sr_p:.2f})",
               edgecolors="black", linewidths=0.8)
# Capital Market Line
vol_range = np.linspace(0, sim_vol.max() * 100 * 1.1, 100)
cml_ret   = RISK_FREE * 252 * 100 + (r_ms * 100 - RISK_FREE * 252 * 100) / (v_ms * 100) * vol_range
ax.plot(vol_range, cml_ret, "k--", linewidth=1.2, label="Capital Market Line")
ax.axhline(RISK_FREE*252*100, color="gray", linewidth=0.7,
           linestyle=":", label=f"Risk-free ({RISK_FREE*252*100:.1f}%)")
ax.set_xlabel("Annualised Volatility (%)")
ax.set_ylabel("Annualised Return (%)")
ax.set_title("Markowitz Efficient Frontier — KSE-30 Constituents")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
savefig("P4_efficient_frontier.png")

# ── P5. Portfolio weights comparison ─────────────────────────────────────────
weight_df = pd.DataFrame({
    "Equal":      w_eq,
    "Market Cap": avg_wt,
    "Min Var":    w_mv,
    "Max Sharpe": w_ms,
}, index=SYMBOLS)
weight_df = weight_df[weight_df.max(axis=1) > 0.001]   # hide near-zero rows

fig, ax = plt.subplots(figsize=(14, 7))
weight_df.T.plot(kind="bar", ax=ax, colormap="tab20", width=0.8)
ax.set_title("Portfolio Weight Allocations by Strategy")
ax.set_ylabel("Weight")
ax.set_xlabel("Strategy")
ax.legend(loc="upper right", ncol=3, fontsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
savefig("P5_weight_comparison_bar.png")

# ── P5b. Weight heatmap ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 12))
sns.heatmap(weight_df * 100, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "Weight (%)"}, linewidths=0.3,
            annot_kws={"fontsize": 7})
ax.set_title("Portfolio Weights (%) by Strategy")
ax.set_xlabel("Strategy")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
savefig("P5b_weight_heatmap.png")

# ── P6. Cumulative returns — all 4 portfolios ─────────────────────────────────
print("\nComputing cumulative portfolio returns …")
r_matrix = returns_wide.values   # (T, N)
dates_p  = returns_wide.index

port_cum = {
    "Equal Weight":  (r_matrix @ w_eq),
    "Market Cap":    (r_matrix @ avg_wt),
    "Min Variance":  (r_matrix @ w_mv),
    "Max Sharpe":    (r_matrix @ w_ms),
}
port_colors = {
    "Equal Weight": "#2ecc71",
    "Market Cap":   "#f39c12",
    "Min Variance": "#3498db",
    "Max Sharpe":   "#e74c3c",
}

fig, ax = plt.subplots(figsize=(13, 6))
for label, daily_r in port_cum.items():
    cum = np.expm1(np.cumsum(daily_r)) * 100   # % total return
    ax.plot(dates_p, cum, label=label, color=port_colors[label], linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title("Cumulative Portfolio Returns — KSE-30 Constituent Portfolios")
ax.set_ylabel("Total Return (%)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("P6_cumulative_returns.png")

# ── P7. Rolling 252-day Sharpe ratio ─────────────────────────────────────────
WINDOW = 252
fig, ax = plt.subplots(figsize=(13, 5))
for label, daily_r in port_cum.items():
    s = pd.Series(daily_r, index=dates_p)
    roll_ret = s.rolling(WINDOW).mean() * 252
    roll_vol = s.rolling(WINDOW).std() * np.sqrt(252)
    roll_sr  = (roll_ret - RISK_FREE * 252) / roll_vol
    ax.plot(dates_p, roll_sr, label=label,
            color=port_colors[label], linewidth=1.3)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title(f"Rolling {WINDOW}-Day Sharpe Ratio by Portfolio Strategy")
ax.set_ylabel("Sharpe Ratio")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("P7_rolling_sharpe.png")

# ── P8. Individual stock return distribution overview ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
violin_data = [returns_wide[sym].dropna().values * 100 for sym in SYMBOLS]
parts = ax.violinplot(violin_data, positions=range(N), showmedians=True,
                      widths=0.7)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
ax.set_xticks(range(N))
ax.set_xticklabels(SYMBOLS, rotation=90, fontsize=7)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title("Daily Return Distributions — KSE-30 Constituents")
ax.set_ylabel("Daily Log Return (%)")
plt.tight_layout()
savefig("P8_return_distributions_violin.png")

# ── P9. Drawdown comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for label, daily_r in port_cum.items():
    cum_series = pd.Series(np.cumsum(daily_r), index=dates_p)
    drawdown   = cum_series - cum_series.cummax()
    ax.fill_between(dates_p, drawdown * 100, 0,
                    alpha=0.35, label=label, color=port_colors[label])
    ax.plot(dates_p, drawdown * 100, color=port_colors[label], linewidth=0.8)
ax.set_title("Portfolio Drawdown Comparison")
ax.set_ylabel("Drawdown (%)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("P9_drawdowns.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 6: Summary tables ─────────────────────────────────────")

# Portfolio performance summary
perf_rows = []
for label, w, daily_r_arr in [
    ("Equal Weight",  w_eq,   r_matrix @ w_eq),
    ("Market Cap",    avg_wt, r_matrix @ avg_wt),
    ("Min Variance",  w_mv,   r_matrix @ w_mv),
    ("Max Sharpe",    w_ms,   r_matrix @ w_ms),
]:
    s    = pd.Series(daily_r_arr)
    ann_ret = s.mean() * 252
    ann_vol = s.std() * np.sqrt(252)
    sr_a    = (ann_ret - RISK_FREE*252) / ann_vol
    max_dd  = (s.cumsum() - s.cumsum().cummax()).min()
    perf_rows.append({
        "Portfolio":          label,
        "Ann. Return (%)":    round(ann_ret*100, 2),
        "Ann. Volatility (%)":round(ann_vol*100, 2),
        "Sharpe Ratio":       round(sr_a, 4),
        "Max Drawdown (%)":   round(max_dd*100, 2),
        "Active stocks":      int((w > 0.001).sum()),
    })

perf_df = pd.DataFrame(perf_rows).set_index("Portfolio")
print("\n[Table] Portfolio Performance Summary")
print(perf_df.to_string())

# Weight output
weight_out = pd.DataFrame({
    "Symbol":              SYMBOLS,
    "Equal Weight (%)":    (w_eq   * 100).round(2),
    "Market Cap Wt (%)":   (avg_wt * 100).round(2),
    "Min Variance Wt (%)": (w_mv   * 100).round(2),
    "Max Sharpe Wt (%)":   (w_ms   * 100).round(2),
    "Ann Return (%)":      (desc.loc[SYMBOLS, "mean_annual"] * 100).round(2).values,
    "Ann Vol (%)":         (desc.loc[SYMBOLS, "vol_annual"]  * 100).round(2).values,
    "Sharpe":              desc.loc[SYMBOLS, "sharpe_annual"].round(4).values,
})
weight_out = weight_out.sort_values("Max Sharpe Wt (%)", ascending=False)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_weights.csv")
weight_out.to_csv(out_path, index=False)
print(f"\nWeights saved → {out_path}")
print(weight_out.head(15).to_string())

print("\n[Table] Top 10 max-Sharpe portfolio stocks:")
top10 = weight_out[weight_out["Max Sharpe Wt (%)"] > 0.1].head(10)
print(top10[["Symbol","Max Sharpe Wt (%)","Ann Return (%)","Ann Vol (%)","Sharpe"]].to_string())

print(f"\nAll figures saved to: {FIG_DIR}")
print("PORTFOLIO OPTIMISATION COMPLETE.")
