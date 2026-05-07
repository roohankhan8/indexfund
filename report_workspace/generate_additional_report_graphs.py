from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
CURSOR_DIR = REPO_ROOT / "6_cursor_model"
CLAUDE_DIR = REPO_ROOT / "5_claude_pipeline"

CH4_IMG = BASE_DIR / "chapter-04-data-collection-and-processing" / "images"
CH7_IMG = BASE_DIR / "chapter-07-discussion" / "images"
CH8_IMG = BASE_DIR / "chapter-08-conclusion-and-recommendations" / "images"

for out_dir in [CH4_IMG, CH7_IMG, CH8_IMG]:
    out_dir.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
    }
)
sns.set_theme(style="whitegrid", palette="muted")


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def load_data():
    daily = pd.read_csv(CURSOR_DIR / "daily_master.csv", parse_dates=["date"])
    monthly = pd.read_csv(CURSOR_DIR / "monthly_master.csv", parse_dates=["date"])
    ff = pd.read_csv(CURSOR_DIR / "results_fund_flow.csv")
    garch = pd.read_csv(CURSOR_DIR / "results_garch.csv")
    eff = pd.read_csv(CURSOR_DIR / "results_efficiency.csv")
    reb = pd.read_csv(CURSOR_DIR / "results_rebalancing.csv")
    forecast = pd.read_csv(CURSOR_DIR / "results_rebalancing_forecast.csv")
    funds = pd.read_excel(CLAUDE_DIR / "funds_data.xlsx", sheet_name=None)
    raw_stock = pd.read_csv(CLAUDE_DIR / "kse30_daily_data.csv")
    return daily, monthly, ff, garch, eff, reb, forecast, funds, raw_stock


def make_ch4_dataset_coverage(funds: dict, raw_stock: pd.DataFrame,
                              daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    spans = []
    spans.append(
        {
            "label": "KSE-30 raw stock file",
            "start": pd.to_datetime(raw_stock["Date"]).min(),
            "end": pd.to_datetime(raw_stock["Date"]).max(),
            "kind": "Market",
        }
    )
    spans.append(
        {
            "label": "Daily master dataset",
            "start": daily["date"].min(),
            "end": daily["date"].max(),
            "kind": "Derived",
        }
    )
    spans.append(
        {
            "label": "Monthly master dataset",
            "start": monthly["date"].min(),
            "end": monthly["date"].max(),
            "kind": "Derived",
        }
    )

    for fund_name, df in funds.items():
        date_col = next(col for col in df.columns if "date" in col.lower())
        dt = pd.to_datetime(df[date_col])
        spans.append(
            {
                "label": f"{fund_name} NAV/AUM",
                "start": dt.min(),
                "end": dt.max(),
                "kind": "Fund",
            }
        )

    span_df = pd.DataFrame(spans).sort_values("start").reset_index(drop=True)
    colors = {"Market": "#1f77b4", "Fund": "#2ca02c", "Derived": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, row in span_df.iterrows():
        ax.barh(
            row["label"],
            (row["end"] - row["start"]).days,
            left=row["start"],
            color=colors[row["kind"]],
            alpha=0.85,
        )
        ax.text(row["end"], idx, f"  {row['end'].date()}", va="center", fontsize=8)

    ax.set_title("Coverage and Overlap of Final Datasets Used in the Study")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=kind)
        for kind, color in colors.items()
    ]
    ax.legend(handles=handles, loc="lower right")
    savefig(fig, CH4_IMG / "C4_01_dataset_coverage_timeline.png")


def make_ch7_flow_scorecard(ff: pd.DataFrame) -> None:
    plot_df = ff.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    rmse_colors = ["#95a5a6", "#e67e22", "#27ae60"]
    axes[0].bar(plot_df["Model"], plot_df["RMSE"], color=rmse_colors, alpha=0.85)
    axes[0].set_title("Aggregate Flow Forecast Error")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=18)
    for i, val in enumerate(plot_df["RMSE"]):
        axes[0].text(i, val + 4, f"{val:.1f}", ha="center", fontsize=8)

    axes[1].bar(plot_df["Model"], plot_df["DirAcc"], color=rmse_colors, alpha=0.85)
    axes[1].axhline(50, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Aggregate Flow Directional Accuracy")
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=18)
    for i, val in enumerate(plot_df["DirAcc"]):
        axes[1].text(i, val + 1.2, f"{val:.1f}%", ha="center", fontsize=8)

    fig.suptitle("KSE-30 Sector Flow Model Scorecard", fontweight="bold")
    plt.tight_layout()
    savefig(fig, CH7_IMG / "C7_01_flow_model_scorecard.png")


def make_ch7_efficiency_summary(eff: pd.DataFrame) -> None:
    row = eff.iloc[0]
    p_df = pd.DataFrame(
        {
            "Test": ["Runs", "Variance Ratio", "Ljung-Box"],
            "p_value": [row["Runs p"], row["VR(2) p"], row["LB Q p"]],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    colors = ["#f39c12", "#2ecc71", "#e74c3c"]
    axes[0].bar(p_df["Test"], p_df["p_value"], color=colors, alpha=0.85)
    axes[0].axhline(0.05, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Efficiency Test p-values")
    axes[0].set_ylabel("p-value")
    axes[0].set_ylim(0, max(0.9, p_df["p_value"].max() + 0.05))
    for i, val in enumerate(p_df["p_value"]):
        axes[0].text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=8)

    hurst = float(row["Hurst H"])
    acf1 = float(row["ACF lag-1"])
    axes[1].bar(["Hurst H", "Lag-1 ACF"], [hurst, acf1], color=["#8e44ad", "#3498db"], alpha=0.85)
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_title("Persistence Indicators")
    axes[1].set_ylabel("Value")
    for i, val in enumerate([hurst, acf1]):
        axes[1].text(i, val + 0.03, f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("KSE-30 Efficiency Evidence Summary", fontweight="bold")
    plt.tight_layout()
    savefig(fig, CH7_IMG / "C7_02_efficiency_evidence_summary.png")


def make_ch7_volatility_regimes(daily: pd.DataFrame) -> None:
    df = daily[["date", "idx_rolling_vol_30d"]].dropna().copy()
    high_cut = df["idx_rolling_vol_30d"].quantile(0.9)
    low_cut = df["idx_rolling_vol_30d"].quantile(0.5)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(df["date"], df["idx_rolling_vol_30d"], color="#2c3e50", linewidth=1.4)
    ax.axhline(high_cut, color="#e74c3c", linestyle="--", linewidth=1.0, label="Top 10% threshold")
    ax.axhline(low_cut, color="#95a5a6", linestyle=":", linewidth=0.9, label="Median")

    high_df = df[df["idx_rolling_vol_30d"] >= high_cut]
    ax.scatter(high_df["date"], high_df["idx_rolling_vol_30d"], color="#e74c3c", s=14, zorder=4)

    ax.set_title("KSE-30 Realized Volatility Regimes (30-day Rolling Annualized Volatility)")
    ax.set_ylabel("Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left")
    savefig(fig, CH7_IMG / "C7_03_realized_volatility_regimes.png")


def make_ch7_model_family_comparison(ff: pd.DataFrame, reb: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    flow_colors = ["#95a5a6", "#e67e22", "#27ae60"]
    axes[0].bar(ff["Model"], ff["DirAcc"], color=flow_colors, alpha=0.85)
    axes[0].set_title("Flow Models\nDirectional Accuracy")
    axes[0].set_ylabel("Percent")
    axes[0].tick_params(axis="x", rotation=18)

    wt = reb[reb["Task"] == "Weight"].copy()
    axes[1].bar(wt["Model"], wt["R2"], color=flow_colors, alpha=0.85)
    axes[1].set_title("Weight Models\nTest R-squared")
    axes[1].tick_params(axis="x", rotation=18)

    inc = reb[reb["Task"] == "Inclusion"].copy()
    auc_vals = []
    for _, row in inc.iterrows():
        if pd.isna(row["AUC"]):
            auc_vals.append(0.5)
        else:
            auc_vals.append(row["AUC"])
    axes[2].bar(inc["Model"], auc_vals, color=flow_colors, alpha=0.85)
    axes[2].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[2].set_title("Inclusion Models\nAUC")
    axes[2].tick_params(axis="x", rotation=18)

    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    fig.suptitle("Model Family Comparison Across Project Tasks", fontweight="bold")
    plt.tight_layout()
    savefig(fig, CH7_IMG / "C7_04_model_family_comparison.png")


def make_ch7_rebalancing_risk_map(forecast: pd.DataFrame) -> None:
    df = forecast.copy()
    df["delta_pred"] = df["pred_wt_avg"] - df["cur_weight"]
    size = np.clip(df["pred_wt_avg"], 0.05, None) * 55

    fig, ax = plt.subplots(figsize=(10.5, 6))
    sc = ax.scatter(
        df["cur_weight"],
        df["exclusion_risk"],
        s=size,
        c=df["delta_pred"],
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.axhline(0.35, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Current KSE-30 Weight")
    ax.set_ylabel("Exclusion Risk")
    ax.set_title("KSE-30 Rebalancing Risk Map")

    label_df = df.sort_values("exclusion_risk", ascending=False).head(8)
    for _, row in label_df.iterrows():
        ax.text(row["cur_weight"] + 0.06, row["exclusion_risk"] + 0.008, row["symbol"], fontsize=8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Predicted Weight Change")
    savefig(fig, CH7_IMG / "C7_05_rebalancing_risk_map.png")


def make_ch8_key_metrics_summary(daily: pd.DataFrame, monthly: pd.DataFrame, ff: pd.DataFrame,
                                 garch: pd.DataFrame, eff: pd.DataFrame, reb: pd.DataFrame) -> None:
    best_flow = ff.sort_values("DirAcc", ascending=False).iloc[0]
    garch_row = garch.iloc[0]
    eff_row = eff.iloc[0]
    best_inclusion = reb[reb["Task"] == "Inclusion"].copy()
    best_inclusion["AUC_fill"] = best_inclusion["AUC"].fillna(0.5)
    best_inclusion = best_inclusion.sort_values("AUC_fill", ascending=False).iloc[0]

    tiles = [
        ("Daily sample", f"{len(daily):,} rows", "#1f77b4"),
        ("Monthly sample", f"{len(monthly):,} rows", "#2ca02c"),
        ("Best flow model", f"{best_flow['Model']}\n{best_flow['DirAcc']:.1f}% DirAcc", "#e67e22"),
        ("Volatility persistence", f"{garch_row['persist']:.4f}", "#d62728"),
        ("Hurst exponent", f"{float(eff_row['Hurst H']):.4f}", "#8e44ad"),
        ("Best inclusion AUC", f"{best_inclusion['Model']}\n{best_inclusion['AUC_fill']:.4f}", "#17becf"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (title, value, color) in zip(axes, tiles):
        ax.set_facecolor(color)
        ax.text(0.5, 0.64, title, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.34, value, ha="center", va="center", color="white",
                fontsize=13, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Final KSE-30 Study Metrics Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig(fig, CH8_IMG / "C8_01_key_metrics_summary.png")


def main() -> None:
    daily, monthly, ff, garch, eff, reb, forecast, funds, raw_stock = load_data()
    make_ch4_dataset_coverage(funds, raw_stock, daily, monthly)
    make_ch7_flow_scorecard(ff)
    make_ch7_efficiency_summary(eff)
    make_ch7_volatility_regimes(daily)
    make_ch7_model_family_comparison(ff, reb)
    make_ch7_rebalancing_risk_map(forecast)
    make_ch8_key_metrics_summary(daily, monthly, ff, garch, eff, reb)


if __name__ == "__main__":
    main()
