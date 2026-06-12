from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "output" / "analysis"
OUT_PATH = (
    BASE_DIR.parent
    / "docs"
    / "report_workspace"
    / "chapter-07-discussion"
    / "images"
    / "C7_05_rebalancing_risk_map_inbubble_labels.png"
)


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)
sns.set_theme(style="whitegrid", palette="muted")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DIR / "results_rebalancing_forecast.csv").copy()
    df["delta_pred"] = df["pred_wt_avg"] - df["cur_weight"]
    size = np.clip(df["pred_wt_avg"], 0.05, None) * 55

    fig, ax = plt.subplots(figsize=(10.5, 6))
    sc = ax.scatter(
        df["cur_weight"],
        df["exclusion_risk"],
        s=size,
        c=df["delta_pred"],
        cmap="coolwarm",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.35,
        zorder=2,
    )

    ax.axhline(0.35, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Current KSE-30 Weight")
    ax.set_ylabel("Exclusion Risk")
    ax.set_title("KSE-30 Rebalancing Risk Map")

    for _, row in df.iterrows():
        bubble_size = max(float(np.clip(row["pred_wt_avg"], 0.05, None) * 55), 1.0)
        font_size = float(np.clip(6.5 + np.sqrt(bubble_size) * 0.13, 6.5, 11.5))
        text_color = "white" if bubble_size >= 300 else "#1f2933"
        ax.text(
            row["cur_weight"],
            row["exclusion_risk"],
            str(row["symbol"]),
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            fontweight="bold",
            zorder=3,
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Predicted Weight Change")

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
