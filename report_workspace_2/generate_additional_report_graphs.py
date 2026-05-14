from pathlib import Path
import shutil
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
CURSOR_DIR = REPO_ROOT / "6_cursor_model"
CLAUDE_DIR = REPO_ROOT / "5_claude_pipeline"

CH3_IMG = BASE_DIR / "chapter-03-methodology" / "images"
CH4_IMG = BASE_DIR / "chapter-04-data-collection-and-processing" / "images"
CH5_IMG = BASE_DIR / "chapter-05-results-and-analysis" / "images"
CH6_IMG = BASE_DIR / "chapter-06-portfolio-tilt-and-rebalancing-application" / "images"
CH7_IMG = BASE_DIR / "chapter-07-discussion" / "images"
CH8_IMG = BASE_DIR / "chapter-08-conclusion-and-recommendations" / "images"
GRAPH_EXPLAIN_DIR = BASE_DIR / "graph_explanations"

for out_dir in [CH3_IMG, CH4_IMG, CH5_IMG, CH6_IMG, CH7_IMG, CH8_IMG, GRAPH_EXPLAIN_DIR]:
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


def write_text(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    print(f"wrote -> {path}")


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


def copy_report_figure(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied -> {dst}")


def sync_existing_report_figures() -> None:
    mappings = [
        (CURSOR_DIR / "figures" / "eda" / "E01_aum_trend.png", CH5_IMG / "E01_aum_trend.png"),
        (CURSOR_DIR / "figures" / "eda" / "E02_nav_return_dist.png", CH5_IMG / "E02_nav_return_dist.png"),
        (CURSOR_DIR / "figures" / "eda" / "E03_fund_flows.png", CH5_IMG / "E03_fund_flows.png"),
        (CURSOR_DIR / "figures" / "eda" / "E05_monthly_correlation.png", CH5_IMG / "E05_monthly_correlation.png"),
        (CURSOR_DIR / "figures" / "eda" / "E06_index_cumulative_return.png", CH5_IMG / "E06_index_cumulative_return.png"),
        (CURSOR_DIR / "figures" / "fund_flow" / "FF01_total_flow_predictions.png", CH5_IMG / "FF01_total_flow_predictions.png"),
        (CURSOR_DIR / "figures" / "fund_flow" / "FF02_granger.png", CH5_IMG / "FF02_granger.png"),
        (CURSOR_DIR / "figures" / "garch" / "G01_returns_and_vol.png", CH5_IMG / "G01_returns_and_vol.png"),
        (CURSOR_DIR / "figures" / "garch" / "G02_var_backtest.png", CH5_IMG / "G02_var_backtest.png"),
        (CURSOR_DIR / "figures" / "efficiency" / "EF01_acf.png", CH5_IMG / "EF01_acf.png"),
        (CURSOR_DIR / "figures" / "efficiency" / "EF02_variance_ratio.png", CH5_IMG / "EF02_variance_ratio.png"),
        (CURSOR_DIR / "figures" / "rebalancing" / "R01_retention_probability.png", CH6_IMG / "R01_retention_probability.png"),
        (CURSOR_DIR / "figures" / "rebalancing" / "R02_feature_importances.png", CH6_IMG / "R02_feature_importances.png"),
        (CURSOR_DIR / "figures" / "rebalancing" / "R03_weight_scatter.png", CH6_IMG / "R03_weight_scatter.png"),
        (CURSOR_DIR / "figures" / "rebalancing" / "R04_weight_changes.png", CH6_IMG / "R04_weight_changes.png"),
    ]
    for src, dst in mappings:
        copy_report_figure(src, dst)


def draw_box(ax, x, y, w, h, title, lines, facecolor):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor="#22313f",
        facecolor=facecolor,
        alpha=0.96,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#10212b")
    ax.text(x + w / 2, y + h * 0.37, "\n".join(lines), ha="center", va="center",
            fontsize=9, color="#10212b")


def connect(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#34495e"),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label, ha="center",
                va="center", fontsize=8, color="#34495e")


def make_ch3_progression_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, "#d9edf7", "4_claude_model", [
            "Modular research scripts",
            "Preprocessing, flow,",
            "GARCH, efficiency,",
            "rebalancing blocks",
        ]),
        (0.375, "#dff0d8", "5_claude_pipeline", [
            "Single-file pipeline",
            "Daily + monthly masters",
            "Unified figures and",
            "results exports",
        ]),
        (0.71, "#fdebd0", "6_cursor_model", [
            "Final KSE-30 focus",
            "Aggregate sector flow",
            "Index-centric volatility",
            "and rebalancing outputs",
        ]),
    ]

    for x, color, title, lines in boxes:
        draw_box(ax, x, 0.24, 0.24, 0.5, title, lines, color)

    connect(ax, 0.28, 0.49, 0.375, 0.49, "merge modules")
    connect(ax, 0.615, 0.49, 0.71, 0.49, "refine to final scope")

    ax.text(0.5, 0.9, "Final Methodological Progression Adopted for the Study",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#10212b")
    ax.text(0.5, 0.1, "The final report should narrate the project as a progression from modular development to a consolidated pipeline and then to a KSE-30 specific final implementation.",
            ha="center", va="center", fontsize=9, color="#455a64")
    savefig(fig, CH3_IMG / "C3_01_methodological_progression.png")


def make_ch3_data_flow_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    input_specs = [
        (0.03, 0.67, "#d6eaf8", "KSE-30 Stock Data", ["Prices", "Weights", "Volume", "Free-float MCAP"]),
        (0.03, 0.39, "#d5f5e3", "Fund NAV and AUM", ["AKD", "NBP", "NTI", "Daily sheets"]),
        (0.03, 0.11, "#fcf3cf", "Macro and CPI", ["Oil", "USD/PKR", "Interest rate", "Monthly CPI"]),
    ]
    for x, y, color, title, lines in input_specs:
        draw_box(ax, x, y, 0.2, 0.18, title, lines, color)

    draw_box(
        ax, 0.33, 0.3, 0.23, 0.38, "5_claude_pipeline", [
            "Clean and align dates",
            "Remove halt rows",
            "Build daily / monthly masters",
            "Compute fund flows",
        ], "#e8daef"
    )
    draw_box(
        ax, 0.66, 0.58, 0.27, 0.2, "Intermediate Outputs", [
            "daily_master.csv",
            "monthly_master.csv",
            "kse30_stocks_clean.csv",
        ], "#f5eef8"
    )
    draw_box(
        ax, 0.66, 0.18, 0.27, 0.26, "6_cursor_model Analytics", [
            "EDA",
            "Flow forecasting",
            "GARCH / EGARCH",
            "Efficiency tests",
            "Rebalancing forecast",
        ], "#fadbd8"
    )

    connect(ax, 0.23, 0.76, 0.33, 0.56)
    connect(ax, 0.23, 0.48, 0.33, 0.48)
    connect(ax, 0.23, 0.20, 0.33, 0.40)
    connect(ax, 0.56, 0.60, 0.66, 0.68, "save")
    connect(ax, 0.56, 0.40, 0.66, 0.31, "feed final study")

    ax.text(0.5, 0.94, "Final Data Pipeline for the KSE-30 Study",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#10212b")
    ax.text(0.79, 0.1, "Folder 5 prepares the masters.\nFolder 6 consumes them for the final KSE-30 analytics.",
            ha="center", va="center", fontsize=9, color="#455a64")
    savefig(fig, CH3_IMG / "C3_02_final_data_flow.png")


def make_ch3_variable_family_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.36, 0.42, 0.28, 0.16, "Final KSE-30 Methodology", [
        "Forecast sector fund flow",
        "Model volatility and efficiency",
        "Predict rebalancing outcomes",
    ], "#eaf2f8")

    specs = [
        (0.08, 0.68, "#d6eaf8", "Target Flow Variable", [
            "total_fund_flow",
            "flow_pct_sector",
            "direction of next-month flow",
        ]),
        (0.66, 0.68, "#d5f5e3", "Macroeconomic Drivers", [
            "interest_rate_end",
            "cpi_yoy_end",
            "oil_return_monthly",
            "usdpkr_return_monthly",
        ]),
        (0.08, 0.14, "#fcf3cf", "KSE-30 Market Variables", [
            "idx_return_monthly",
            "idx_vol_monthly",
            "daily index returns",
            "conditional volatility",
        ]),
        (0.66, 0.14, "#fadbd8", "Rebalancing Features", [
            "weight, drift, range",
            "momentum 30 / 60 / 90",
            "volatility, volume, MA gaps",
        ]),
    ]
    anchors = [(0.36, 0.58), (0.64, 0.58), (0.36, 0.42), (0.64, 0.42)]
    for (x, y, color, title, lines), (ax_x, ax_y) in zip(specs, anchors):
        draw_box(ax, x, y, 0.24, 0.18, title, lines, color)
        connect(ax, x + 0.12, y if y > 0.5 else y + 0.18, ax_x, ax_y)

    ax.text(0.5, 0.94, "Main Variable Families Used in the Final KSE-30 Methodology",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#10212b")
    savefig(fig, CH3_IMG / "C3_03_variable_families.png")


def make_ch6_rebalancing_framework_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, "#d6eaf8", "Historical KSE-30 Panel", [
            "Constituent history",
            "Weights, returns,",
            "volume and prices",
        ]),
        (0.27, "#d5f5e3", "Detect Rebalancing Dates", [
            "Semi-annual windows",
            "Pre-event snapshots",
            "Next-period targets",
        ]),
        (0.51, "#fcf3cf", "Build Features", [
            "Momentum, volatility,",
            "liquidity, moving averages,",
            "weight drift and range",
        ]),
        (0.75, "#fadbd8", "Predict Outcomes", [
            "Next weight",
            "Retention probability",
            "Portfolio tilt signal",
        ]),
    ]

    for x, color, title, lines in boxes:
        draw_box(ax, x, 0.27, 0.19, 0.42, title, lines, color)
    connect(ax, 0.22, 0.48, 0.27, 0.48)
    connect(ax, 0.46, 0.48, 0.51, 0.48)
    connect(ax, 0.70, 0.48, 0.75, 0.48)

    ax.text(0.5, 0.88, "KSE-30 Rebalancing Prediction Workflow",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#10212b")
    savefig(fig, CH6_IMG / "C6_01_rebalancing_framework.png")


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
    axes[1].bar(["Hurst H", "Lag-1 ACF"], [hurst, acf1],
                color=["#8e44ad", "#3498db"], alpha=0.85)
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
    auc_vals = inc["AUC"].fillna(0.5).tolist()
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


def write_graph_explanations(daily: pd.DataFrame, monthly: pd.DataFrame, ff: pd.DataFrame,
                             garch: pd.DataFrame, eff: pd.DataFrame,
                             reb: pd.DataFrame, forecast: pd.DataFrame) -> None:
    flow_min = monthly.loc[monthly["total_fund_flow"].idxmin()]
    flow_max = monthly.loc[monthly["total_fund_flow"].idxmax()]
    best_flow = ff.sort_values("DirAcc", ascending=False).iloc[0]
    garch_row = garch.iloc[0]
    eff_row = eff.iloc[0]
    weight_best = reb[reb["Task"] == "Weight"].sort_values("R2", ascending=False).iloc[0]
    inc_df = reb[reb["Task"] == "Inclusion"].copy()
    inc_df["AUC_fill"] = inc_df["AUC"].fillna(0.5)
    inclusion_best = inc_df.sort_values("AUC_fill", ascending=False).iloc[0]
    highest_risk = forecast.sort_values("exclusion_risk", ascending=False).iloc[0]

    write_text(
        GRAPH_EXPLAIN_DIR / "README.md",
        """
        # Graph Explanations

        This folder contains chapter-wise notes for every figure currently allocated to the report.

        Use these notes when:
        - writing figure captions
        - explaining the meaning of a graph in the viva or presentation
        - keeping interpretation consistent across chapters

        The paths mentioned below refer to the chapter image copies inside `report_workspace/`.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-03-explanations.md",
        f"""
        # Chapter 3 Graph Explanations

        ## Figure 3.1
        Path: `report_workspace/chapter-03-methodology/images/C3_01_methodological_progression.png`

        - This diagram shows how the final methodology evolved from `4_claude_model/` to `5_claude_pipeline/` and finally to `6_cursor_model/`.
        - Use it to explain that the dissertation does not treat the project as one sudden final script. It presents a clear methodological progression.
        - The main interpretation is that folder 4 contributed modular research logic, folder 5 consolidated the workflow, and folder 6 became the final KSE-30 specific implementation.

        ## Figure 3.2
        Path: `report_workspace/chapter-03-methodology/images/C3_02_final_data_flow.png`

        - This figure shows the data pipeline from raw stock, fund, macro, and CPI inputs into the cleaned master datasets and then into the final KSE-30 analytics.
        - Use it when describing how folder 5 prepares `daily_master.csv`, `monthly_master.csv`, and `kse30_stocks_clean.csv`, while folder 6 consumes those outputs.
        - The key message is that preprocessing and final analytics are separated but connected in one reproducible workflow.

        ## Figure 3.3
        Path: `report_workspace/chapter-03-methodology/images/C3_03_variable_families.png`

        - This diagram groups the study variables into four families: target flow variables, macroeconomic drivers, KSE-30 market variables, and rebalancing features.
        - Use it to show that the final methodology is not only a forecasting exercise. It combines fund, market, macro, and stock-level signals in one design.
        - The interpretation should emphasize that each variable family supports a different analytical block of the project.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-04-explanations.md",
        f"""
        # Chapter 4 Graph Explanations

        ## Figure 4.1
        Path: `report_workspace/chapter-04-data-collection-and-processing/images/C4_01_dataset_coverage_timeline.png`

        - This timeline shows the date coverage and overlap of the raw stock file, the three fund sheets, and the final daily and monthly master datasets.
        - Use it to justify why the empirical analysis starts only where the fund, stock, and macro series overlap reliably.
        - The graph supports the statement that the final daily dataset has {len(daily):,} rows and the final monthly dataset has {len(monthly):,} observations.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-05-explanations.md",
        f"""
        # Chapter 5 Graph Explanations

        ## Figure 5.1
        Path: `report_workspace/chapter-05-results-and-analysis/images/E06_index_cumulative_return.png`

        - This graph shows the cumulative return path of the reconstructed KSE-30 index over the final analysis window.
        - Use it to give the reader a high-level picture of market direction before discussing volatility and efficiency.

        ## Figure 5.2
        Path: `report_workspace/chapter-05-results-and-analysis/images/E01_aum_trend.png`

        - This line chart shows the combined sector AUM of the KSE-30 related funds over time.
        - It is mainly used to show concentration and growth in the tracked fund universe.
        - A raw data gap in May 2024 had all three funds recorded at zero AUM. The pipeline now repairs that isolated month before plotting, so the figure should be read as a cleaned trend rather than a literal liquidation event.

        ## Figure 5.3
        Path: `report_workspace/chapter-05-results-and-analysis/images/E02_nav_return_dist.png`

        - This distribution plot shows that daily returns are non-normal and fat-tailed.
        - Use it to support the later choice of volatility models instead of assuming smooth Gaussian behavior.

        ## Figure 5.4
        Path: `report_workspace/chapter-05-results-and-analysis/images/E03_fund_flows.png`

        - This bar chart shows monthly aggregate KSE-30 sector net flow, separating inflow and outflow months visually.
        - After the AUM repair, the largest negative observed flow month in the final monthly master is {flow_min['date'].date()} with approximately {flow_min['total_fund_flow']:.2f} PKR million, while the strongest positive month is {flow_max['date'].date()} with approximately {flow_max['total_fund_flow']:.2f} PKR million.
        - Use this graph to explain that the sector flow series is episodic and shock-prone even after data cleaning.

        ## Figure 5.5
        Path: `report_workspace/chapter-05-results-and-analysis/images/E05_monthly_correlation.png`

        - This heatmap summarizes contemporaneous monthly correlations between sector flow, macro variables, and KSE-30 market measures.
        - Use it to argue that simple same-period linear relationships are weak, which motivates lagged time-series modelling.

        ## Figure 5.6
        Path: `report_workspace/chapter-05-results-and-analysis/images/FF02_granger.png`

        - This graph summarizes Granger-causality p-values for the selected macro variables against aggregate flow.
        - Its purpose is to show that no single macro variable dominates the monthly flow process at lag 1.

        ## Figure 5.7
        Path: `report_workspace/chapter-05-results-and-analysis/images/FF01_total_flow_predictions.png`

        - This figure compares actual aggregate flow against the ARIMAX and VAR predictions.
        - Use it to show that the final models are better interpreted as directional tools rather than exact point estimators.
        - In the cleaned final run, ARIMAX has the lower RMSE, while both retained dynamic models reach {best_flow['DirAcc']:.1f}% directional accuracy.

        ## Figure 5.8
        Path: `report_workspace/chapter-05-results-and-analysis/images/G01_returns_and_vol.png`

        - This chart overlays KSE-30 returns with model-implied conditional volatility.
        - It is used to illustrate volatility clustering and the persistence of high-risk episodes.

        ## Figure 5.9
        Path: `report_workspace/chapter-05-results-and-analysis/images/G02_var_backtest.png`

        - This figure shows the Value-at-Risk backtest for the preferred volatility specification.
        - Use it to explain whether the model's downside risk envelope is broadly calibrated to actual tail events.

        ## Figure 5.10
        Path: `report_workspace/chapter-05-results-and-analysis/images/EF02_variance_ratio.png`

        - This graph visualizes the variance-ratio evidence across horizons or sub-periods.
        - It supports the claim that some efficiency tests are closer to random-walk behavior than others.

        ## Figure 5.11
        Path: `report_workspace/chapter-05-results-and-analysis/images/EF01_acf.png`

        - This autocorrelation plot highlights whether the reconstructed KSE-30 return series contains serial structure across lags.
        - Use it together with the Ljung-Box and Hurst results when discussing mixed efficiency evidence.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-06-explanations.md",
        f"""
        # Chapter 6 Graph Explanations

        ## Figure 6.1
        Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/C6_01_rebalancing_framework.png`

        - This diagram summarizes the stock-level rebalancing workflow from constituent history to pre-event feature windows and finally to predicted weight and retention outcomes.
        - Use it to explain that the rebalancing block is an application layer built on top of the earlier market and flow analysis.

        ## Figure 6.2
        Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R02_feature_importances.png`

        - This figure shows which rebalancing predictors contribute most strongly to the retained model family.
        - Use it to support the discussion that current weight, size-related measures, and stability features matter more than short-run momentum alone.

        ## Figure 6.3
        Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R03_weight_scatter.png`

        - This scatter plot compares observed and predicted constituent weights in the test sample.
        - Use it to show that weight persistence is strong and that the preferred model tracks larger weights reasonably well.
        - The best weight model retained in the results file is {weight_best['Model']} with test R-squared of {weight_best['R2']:.4f}.

        ## Figure 6.4
        Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R01_retention_probability.png`

        - This figure shows predicted retention probabilities across stocks in the test or forecast setting.
        - Use it to explain why AUC is more informative than raw accuracy in a heavily imbalanced retention problem.
        - The best retained inclusion model is {inclusion_best['Model']} with AUC of {inclusion_best['AUC_fill']:.4f}.

        ## Figure 6.5
        Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R04_weight_changes.png`

        - This graph shows predicted weight changes across the forecast universe.
        - Use it to identify likely gainers, likely decliners, and names with near-zero target weight.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-07-explanations.md",
        f"""
        # Chapter 7 Graph Explanations

        ## Figure 7.1
        Path: `report_workspace/chapter-07-discussion/images/C7_01_flow_model_scorecard.png`

        - This scorecard compares the final aggregate-flow models on forecast error and directional accuracy.
        - Use it to support the interpretation that the best model is more useful directionally than in exact PKR magnitude terms.

        ## Figure 7.2
        Path: `report_workspace/chapter-07-discussion/images/C7_02_efficiency_evidence_summary.png`

        - This figure condenses the mixed efficiency evidence into one discussion-friendly visual.
        - Use it to explain why the KSE-30 cannot be described as perfectly efficient or perfectly inefficient.
        - The current full-sample Hurst estimate is {float(eff_row['Hurst H']):.4f}.

        ## Figure 7.3
        Path: `report_workspace/chapter-07-discussion/images/C7_03_realized_volatility_regimes.png`

        - This plot highlights periods when realized volatility rises into the upper tail of its own distribution.
        - Use it to discuss volatility clustering and the persistence of elevated risk states.

        ## Figure 7.4
        Path: `report_workspace/chapter-07-discussion/images/C7_04_model_family_comparison.png`

        - This comparison figure places econometric and machine-learning style models side by side across the different project tasks.
        - Use it to argue that model choice should depend on data structure and task granularity.

        ## Figure 7.5
        Path: `report_workspace/chapter-07-discussion/images/C7_05_rebalancing_risk_map.png`

        - This risk map relates current index weight to exclusion risk and predicted weight change.
        - It is useful for discussing portfolio tilt because it separates core names from vulnerable names visually.
        - The current highest-risk forecasted name is {highest_risk['symbol']} with exclusion risk of {highest_risk['exclusion_risk']:.2%}.
        """,
    )

    write_text(
        GRAPH_EXPLAIN_DIR / "chapter-08-explanations.md",
        f"""
        # Chapter 8 Graph Explanations

        ## Figure 8.1
        Path: `report_workspace/chapter-08-conclusion-and-recommendations/images/C8_01_key_metrics_summary.png`

        - This figure acts as a closing dashboard for the study.
        - Use it to summarize sample size, the strongest retained forecasting result, volatility persistence, efficiency evidence, and rebalancing performance in one place.
        - The current volatility persistence estimate in the results file is {garch_row['persist']:.4f}, which is one of the main takeaways of the project.
        """,
    )


def main() -> None:
    daily, monthly, ff, garch, eff, reb, forecast, funds, raw_stock = load_data()
    sync_existing_report_figures()
    make_ch3_progression_diagram()
    make_ch3_data_flow_diagram()
    make_ch3_variable_family_diagram()
    make_ch6_rebalancing_framework_diagram()
    make_ch4_dataset_coverage(funds, raw_stock, daily, monthly)
    make_ch7_flow_scorecard(ff)
    make_ch7_efficiency_summary(eff)
    make_ch7_volatility_regimes(daily)
    make_ch7_model_family_comparison(ff, reb)
    make_ch7_rebalancing_risk_map(forecast)
    make_ch8_key_metrics_summary(daily, monthly, ff, garch, eff, reb)
    write_graph_explanations(daily, monthly, ff, garch, eff, reb, forecast)


if __name__ == "__main__":
    main()
