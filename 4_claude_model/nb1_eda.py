"""
NOTEBOOK 1 — Exploratory Data Analysis
========================================
Inputs : daily_master.csv, monthly_master.csv
Outputs: figures saved to  ./figures/eda/
Run    : python nb1_eda.py

Covers
  A. Daily data  — macro series, fund NAV returns, volatility, cross-correlations
  B. Monthly data — fund flows, AUM trends, macro vs flow relationships
  C. Statistical summaries printed to console (copy into thesis tables)
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

# ── output directory ────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "eda")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

# ── style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

FUND_COLORS = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NIT": "#2ca02c"}

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data …")
daily   = pd.read_csv("new_data/daily_master.csv",   parse_dates=["date"])
monthly = pd.read_csv("new_data/monthly_master.csv", parse_dates=["date"])
daily   = daily.sort_values("date").reset_index(drop=True)
monthly = monthly.sort_values("date").reset_index(drop=True)

print(f"Daily   : {daily.shape[0]:,} rows  {daily['date'].min().date()} → {daily['date'].max().date()}")
print(f"Monthly : {monthly.shape[0]:,} rows  {monthly['date'].min().date()} → {monthly['date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════
# A. DAILY DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n── A. Daily EDA ──────────────────────────────────────────────────")

# ── A1. Macro time-series panel ──────────────────────────────────────────────
print("A1. Macro time-series panel …")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Daily Macro Indicators — KSE-30 Analysis Window (2021–2025)", fontweight="bold")

ax = axes[0]
ax.plot(daily["date"], daily["oil_price"], color="#c0392b", linewidth=1)
ax.set_ylabel("Brent Oil (USD/bbl)")
ax.set_title("Brent Crude Oil Price")

ax = axes[1]
ax.plot(daily["date"], daily["usdpkr"], color="#8e44ad", linewidth=1)
ax.set_ylabel("PKR per USD")
ax.set_title("USD/PKR Exchange Rate")

ax = axes[2]
ax.step(daily["date"], daily["interest_rate"], color="#16a085", linewidth=1.5, where="post")
ax.set_ylabel("Rate (%)")
ax.set_title("SBP Policy Interest Rate (forward-filled between decisions)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("A1_macro_daily.png")

# ── A2. Fund NAV levels ──────────────────────────────────────────────────────
print("A2. Fund NAV levels …")
fig, ax = plt.subplots(figsize=(13, 5))
for fund, col in [("AKD","nav_akd"), ("NBP","nav_nbp"), ("NIT","nav_nti")]:
    ax.plot(daily["date"], daily[col], label=fund,
            color=FUND_COLORS[fund], linewidth=1.2)
ax.set_title("Daily NAV — AKD, NBP, NIT (Jan 2021 – Oct 2025)")
ax.set_ylabel("NAV (PKR per unit)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
savefig("A2_nav_levels.png")

# ── A3. Fund NAV log returns — distribution ──────────────────────────────────
print("A3. NAV return distributions …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Daily NAV Log Return Distributions", fontweight="bold")
for i, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                   ("NBP","nav_return_nbp"),
                                   ("NIT","nav_return_nti")]):
    s = daily[col].dropna()
    ax = axes[i]
    ax.hist(s, bins=60, color=FUND_COLORS[fund], alpha=0.75, edgecolor="white",
            density=True, label="Observed")
    # Overlay normal fit
    mu, sigma = s.mean(), s.std()
    x = np.linspace(s.min(), s.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", linewidth=1.2, label="Normal fit")
    skew, kurt = s.skew(), s.kurtosis()
    ax.set_title(f"{fund}\nμ={mu:.4f}  σ={sigma:.4f}\nskew={skew:.2f}  excess kurt={kurt:.2f}")
    ax.set_xlabel("Log return")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("A3_nav_return_dist.png")

# ── A4. Rolling 30-day volatility ────────────────────────────────────────────
print("A4. Rolling volatility …")
fig, ax = plt.subplots(figsize=(13, 5))
for fund, col in [("AKD","nav_vol_akd"), ("NBP","nav_vol_nbp"), ("NIT","nav_vol_nti")]:
    ax.plot(daily["date"], daily[col], label=fund,
            color=FUND_COLORS[fund], linewidth=1, alpha=0.85)
ax.set_title("Annualised 30-Day Rolling Volatility of Fund NAV Returns")
ax.set_ylabel("Annualised Volatility")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
savefig("A4_rolling_volatility.png")

# ── A5. Macro vs NAV return scatter grid ─────────────────────────────────────
print("A5. Macro vs NAV return scatter …")
macro_vars = {
    "oil_log_return":    "Brent Oil Log Return",
    "usdpkr_log_return": "USD/PKR Log Return",
}
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Daily Macro Returns vs Fund NAV Returns", fontweight="bold")
for row, (macro_col, macro_label) in enumerate(macro_vars.items()):
    for col_i, (fund, nav_col) in enumerate([("AKD","nav_return_akd"),
                                              ("NBP","nav_return_nbp"),
                                              ("NIT","nav_return_nti")]):
        ax = axes[row][col_i]
        x = daily[macro_col].dropna()
        merged = daily[[macro_col, nav_col]].dropna()
        corr = merged.corr().iloc[0, 1]
        ax.scatter(merged[macro_col], merged[nav_col], alpha=0.3,
                   s=6, color=FUND_COLORS[fund])
        # Regression line
        m, b = np.polyfit(merged[macro_col], merged[nav_col], 1)
        xl = np.linspace(merged[macro_col].min(), merged[macro_col].max(), 100)
        ax.plot(xl, m * xl + b, "k--", linewidth=1)
        ax.set_xlabel(macro_label, fontsize=8)
        ax.set_ylabel(f"{fund} NAV return", fontsize=8)
        ax.set_title(f"r = {corr:.3f}", fontsize=10)
plt.tight_layout()
savefig("A5_macro_vs_nav_scatter.png")

# ── A6. Correlation heatmap — daily ─────────────────────────────────────────
print("A6. Daily correlation heatmap …")
corr_cols_daily = [
    "oil_log_return", "usdpkr_log_return",
    "nav_return_akd", "nav_return_nbp", "nav_return_nti",
]
labels_daily = [
    "Oil return", "USD/PKR return",
    "AKD NAV ret", "NBP NAV ret", "NIT NAV ret",
]
corr_d = daily[corr_cols_daily].dropna().corr()
corr_d.index   = labels_daily
corr_d.columns = labels_daily

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_d, dtype=bool), k=1)
sns.heatmap(corr_d, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, mask=mask,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Daily Returns & Macro Indicators")
plt.tight_layout()
savefig("A6_daily_correlation_heatmap.png")

# ── A7. Volatility clustering — Ljung-Box style visual ───────────────────────
print("A7. Squared returns (volatility clustering evidence) …")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Squared NAV Returns — Volatility Clustering Evidence", fontweight="bold")
for i, (fund, col) in enumerate([("AKD","nav_return_akd"),
                                   ("NBP","nav_return_nbp"),
                                   ("NIT","nav_return_nti")]):
    sq = daily[col].dropna() ** 2
    dates = daily.loc[daily[col].notna(), "date"]
    axes[i].fill_between(dates, sq, alpha=0.6, color=FUND_COLORS[fund])
    axes[i].set_ylabel("Return²")
    axes[i].set_title(f"{fund} — Squared Daily NAV Returns")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("A7_squared_returns_clustering.png")

# ── A8. Interest rate overlay with NAV ───────────────────────────────────────
print("A8. Interest rate vs NAV …")
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
for fund, col in [("AKD","nav_akd"), ("NIT","nav_nti")]:
    ax1.plot(daily["date"], daily[col], color=FUND_COLORS[fund],
             linewidth=1, alpha=0.8, label=f"{fund} NAV")
ax2.step(daily["date"], daily["interest_rate"], color="#c0392b",
         linewidth=2, where="post", linestyle="--", label="Interest Rate")
ax1.set_ylabel("NAV (PKR)")
ax2.set_ylabel("SBP Rate (%)", color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
ax1.set_title("Fund NAV vs SBP Policy Rate")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=30)
savefig("A8_interest_rate_vs_nav.png")


# ═══════════════════════════════════════════════════════════════════════════
# B. MONTHLY DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n── B. Monthly EDA ────────────────────────────────────────────────")

# ── B1. AUM over time ────────────────────────────────────────────────────────
print("B1. AUM over time …")
fig, ax = plt.subplots(figsize=(13, 5))
for fund, col in [("AKD","aum_akd"), ("NBP","aum_nbp"), ("NIT","aum_nti")]:
    ax.plot(monthly["date"], monthly[col], marker="o", markersize=4,
            label=fund, color=FUND_COLORS[fund], linewidth=1.5)
ax.set_title("Monthly AUM — AKD, NBP, NIT (PKR Millions)")
ax.set_ylabel("AUM (PKR mn)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
savefig("B1_aum_over_time.png")

# ── B2. Fund flows bar charts ─────────────────────────────────────────────────
print("B2. Fund flows bar charts …")
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Monthly Fund Flows (PKR Millions) — Positive = Inflow, Negative = Outflow",
             fontweight="bold")
for i, (fund, col, spike_col) in enumerate([
    ("AKD", "flow_akd",  "flow_spike_akd"),
    ("NBP", "flow_nbp",  "flow_spike_nbp"),
    ("NIT", "flow_nti",  "flow_spike_nti"),
]):
    ax = axes[i]
    colors = [FUND_COLORS[fund] if v >= 0 else "#e74c3c"
              for v in monthly[col]]
    bars = ax.bar(monthly["date"], monthly[col], color=colors,
                  width=20, edgecolor="white", linewidth=0.4)
    # Mark spikes
    spikes = monthly[monthly[spike_col]]
    ax.scatter(spikes["date"], spikes[col], color="black",
               zorder=5, s=40, marker="*", label="Flow spike (>2σ)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Flow (PKR mn)")
    ax.set_title(f"{fund} Monthly Fund Flow")
    if spikes.shape[0] > 0:
        ax.legend()
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("B2_fund_flows_bar.png")

# ── B3. Total fund flow vs CPI & interest rate ───────────────────────────────
print("B3. Total fund flow vs macro …")
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Total Combined Fund Flow vs Macro Conditions", fontweight="bold")

ax1 = axes[0]
ax2 = ax1.twinx()
ax1.bar(monthly["date"], monthly["total_fund_flow"],
        color=["#2980b9" if v >= 0 else "#e74c3c" for v in monthly["total_fund_flow"]],
        width=20, alpha=0.8, label="Total fund flow")
ax2.plot(monthly["date"], monthly["cpi_yoy_end"], color="#8e44ad",
         linewidth=2, marker="o", markersize=4, label="CPI YoY %")
ax1.set_ylabel("Total Flow (PKR mn)")
ax2.set_ylabel("CPI YoY %", color="#8e44ad")
ax2.tick_params(axis="y", labelcolor="#8e44ad")
ax1.axhline(0, color="black", linewidth=0.5)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("Total Fund Flow vs CPI (Inflation)")

ax1b = axes[1]
ax2b = ax1b.twinx()
ax1b.bar(monthly["date"], monthly["total_fund_flow"],
         color=["#2980b9" if v >= 0 else "#e74c3c" for v in monthly["total_fund_flow"]],
         width=20, alpha=0.8, label="Total fund flow")
ax2b.step(monthly["date"], monthly["interest_rate_end"], color="#c0392b",
          linewidth=2, where="post", label="Interest Rate %")
ax1b.set_ylabel("Total Flow (PKR mn)")
ax2b.set_ylabel("Rate %", color="#c0392b")
ax2b.tick_params(axis="y", labelcolor="#c0392b")
ax1b.axhline(0, color="black", linewidth=0.5)
lines1b, labels1b = ax1b.get_legend_handles_labels()
lines2b, labels2b = ax2b.get_legend_handles_labels()
ax1b.legend(lines1b + lines2b, labels1b + labels2b, loc="upper left")
ax1b.set_title("Total Fund Flow vs SBP Policy Rate")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
savefig("B3_flow_vs_macro.png")

# ── B4. Monthly NAV return comparison ────────────────────────────────────────
print("B4. Monthly NAV returns …")
fig, ax = plt.subplots(figsize=(13, 5))
width = 7
offsets = {"AKD": -width, "NBP": 0, "NIT": width}
for fund, col in [("AKD","nav_return_akd_monthly"),
                   ("NBP","nav_return_nbp_monthly"),
                   ("NIT","nav_return_nti_monthly")]:
    ax.bar(monthly["date"] + pd.Timedelta(days=offsets[fund]),
           monthly[col] * 100,
           width=width - 1, color=FUND_COLORS[fund],
           alpha=0.85, label=fund)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Monthly NAV Returns by Fund (%)")
ax.set_ylabel("NAV Return (%)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
fig.autofmt_xdate(rotation=30)
savefig("B4_monthly_nav_returns.png")

# ── B5. Monthly correlation heatmap ──────────────────────────────────────────
print("B5. Monthly correlation heatmap …")
corr_cols_m = [
    "oil_return_monthly", "usdpkr_return_monthly",
    "interest_rate_end", "cpi_yoy_end",
    "nav_return_akd_monthly", "nav_return_nbp_monthly", "nav_return_nti_monthly",
    "flow_pct_akd", "flow_pct_nbp", "flow_pct_nti",
]
labels_m = [
    "Oil return", "USD/PKR return",
    "Interest rate", "CPI YoY",
    "AKD NAV ret", "NBP NAV ret", "NIT NAV ret",
    "AKD flow %", "NBP flow %", "NIT flow %",
]
corr_m = monthly[corr_cols_m].dropna().corr()
corr_m.index   = labels_m
corr_m.columns = labels_m

fig, ax = plt.subplots(figsize=(12, 9))
mask_m = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, mask=mask_m,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 8})
ax.set_title("Monthly Correlation Matrix — Returns, Macro & Fund Flows")
plt.tight_layout()
savefig("B5_monthly_correlation_heatmap.png")

# ── B6. Cross-correlation: fund flow vs macro (leads/lags) ───────────────────
print("B6. Cross-correlation: fund flow lag analysis …")
lag_range = range(-6, 7)
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle("Cross-Correlation: Fund Flow % vs Macro Variables\n"
             "(negative lag = flow leads macro, positive = macro leads flow)",
             fontweight="bold")
pairs = [
    ("flow_pct_akd", "cpi_yoy_end",        "AKD flow % vs CPI"),
    ("flow_pct_akd", "interest_rate_end",   "AKD flow % vs Interest Rate"),
    ("flow_pct_nbp", "cpi_yoy_end",         "NBP flow % vs CPI"),
    ("flow_pct_nbp", "interest_rate_end",   "NBP flow % vs Interest Rate"),
    ("flow_pct_nti", "cpi_yoy_end",         "NIT flow % vs CPI"),
    ("flow_pct_nti", "interest_rate_end",   "NIT flow % vs Interest Rate"),
]
for ax, (flow_col, macro_col, title) in zip(axes.flat, pairs):
    flow  = monthly[flow_col].dropna().values
    macro = monthly[macro_col].dropna().values
    n = min(len(flow), len(macro))
    flow, macro = flow[-n:], macro[-n:]
    xcorrs = []
    for lag in lag_range:
        if lag < 0:
            c = np.corrcoef(flow[-lag:], macro[:lag])[0, 1]
        elif lag == 0:
            c = np.corrcoef(flow, macro)[0, 1]
        else:
            c = np.corrcoef(flow[:-lag], macro[lag:])[0, 1]
        xcorrs.append(c)
    ax.bar(lag_range, xcorrs,
           color=["#2980b9" if v >= 0 else "#e74c3c" for v in xcorrs])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline( 2 / np.sqrt(n), color="gray", linewidth=1, linestyle="--")
    ax.axhline(-2 / np.sqrt(n), color="gray", linewidth=1, linestyle="--",
               label="95% CI")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=7)
plt.tight_layout()
savefig("B6_cross_correlation_flow_macro.png")

# ── B7. Fund flow % scatter vs macro ─────────────────────────────────────────
print("B7. Fund flow scatter vs CPI …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Monthly Fund Flow (% of AUM) vs CPI YoY Inflation", fontweight="bold")
for i, (fund, col) in enumerate([("AKD","flow_pct_akd"),
                                   ("NBP","flow_pct_nbp"),
                                   ("NIT","flow_pct_nti")]):
    ax = axes[i]
    merged = monthly[[col, "cpi_yoy_end"]].dropna()
    corr = merged.corr().iloc[0, 1]
    ax.scatter(merged["cpi_yoy_end"], merged[col] * 100,
               alpha=0.7, color=FUND_COLORS[fund], s=50)
    m, b = np.polyfit(merged["cpi_yoy_end"], merged[col] * 100, 1)
    xl = np.linspace(merged["cpi_yoy_end"].min(), merged["cpi_yoy_end"].max(), 100)
    ax.plot(xl, m * xl + b, "k--", linewidth=1)
    ax.set_title(f"{fund}  (r = {corr:.3f})")
    ax.set_xlabel("CPI YoY %")
    ax.set_ylabel("Fund flow (% of AUM)")
plt.tight_layout()
savefig("B7_flow_vs_cpi_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# C. STATISTICAL SUMMARIES (for thesis tables)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── C. Printed summaries ──────────────────────────────────────────")

print("\n[Table] Descriptive statistics — Daily NAV returns")
desc_cols = ["nav_return_akd", "nav_return_nbp", "nav_return_nti"]
desc = daily[desc_cols].describe().T
desc.index = ["AKD", "NBP", "NIT"]
desc["skewness"] = [daily[c].skew()     for c in desc_cols]
desc["kurtosis"] = [daily[c].kurtosis() for c in desc_cols]
# Jarque-Bera
for fund, col in zip(["AKD","NBP","NIT"], desc_cols):
    jb_stat, jb_p = stats.jarque_bera(daily[col].dropna())
    print(f"  {fund} Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.4f}  "
          f"({'Non-normal' if jb_p < 0.05 else 'Normal'})")
print(desc[["mean","std","min","max","skewness","kurtosis"]].round(6).to_string())

print("\n[Table] Descriptive statistics — Monthly fund flows (% of AUM)")
flow_cols_pct = ["flow_pct_akd", "flow_pct_nbp", "flow_pct_nti"]
flow_desc = monthly[flow_cols_pct].describe().T
flow_desc.index = ["AKD", "NBP", "NIT"]
print(flow_desc.round(4).to_string())

print("\n[Table] Inflow vs outflow months")
for fund, col in zip(["AKD","NBP","NIT"], flow_cols_pct):
    inflow  = (monthly[col] > 0).sum()
    outflow = (monthly[col] < 0).sum()
    print(f"  {fund}: {inflow} inflow months, {outflow} outflow months "
          f"({inflow/(inflow+outflow)*100:.0f}% inflow rate)")

print("\n[Table] Pairwise Pearson correlations — Monthly")
print(corr_m.round(3).to_string())

print("\n[Summary] Cross-correlation peak lags (fund flow % → macro)")
for flow_col, macro_col, title in pairs:
    flow  = monthly[flow_col].dropna().values
    macro = monthly[macro_col].dropna().values
    n = min(len(flow), len(macro))
    flow, macro = flow[-n:], macro[-n:]
    xcorrs = []
    for lag in lag_range:
        if lag < 0:
            c = np.corrcoef(flow[-lag:], macro[:lag])[0, 1]
        elif lag == 0:
            c = np.corrcoef(flow, macro)[0, 1]
        else:
            c = np.corrcoef(flow[:-lag], macro[lag:])[0, 1]
        xcorrs.append(c)
    best_lag = list(lag_range)[np.argmax(np.abs(xcorrs))]
    best_corr = xcorrs[np.argmax(np.abs(xcorrs))]
    print(f"  {title}: peak r={best_corr:.3f} at lag={best_lag:+d} months")

print(f"\nAll EDA figures saved to: {FIG_DIR}")
print("EDA COMPLETE.")
