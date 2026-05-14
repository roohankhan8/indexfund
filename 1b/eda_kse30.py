import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = (
    Path(__file__).parent.parent / "0-raw-data" / "csvs" / "kse30_daily_data.csv"
)
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["FF BASED SHARES", "FF BASED MCAP", "ORD SHARES", "ORD SHARES MCAP"]:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    return df


def basic_info(df):
    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nUnique symbols: {df['SYMBOL'].nunique()}")
    print(f"Unique dates: {df['Date'].nunique()}")


def missing_values(df):
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_pct})
    print(
        missing_df[missing_df["Missing"] > 0]
        if missing.sum() > 0
        else "No missing values"
    )


def numeric_summary(df):
    print("\n" + "=" * 60)
    print("NUMERIC SUMMARY")
    print("=" * 60)
    numeric_cols = [
        "PRICE",
        "IDX WT %",
        "FF BASED SHARES",
        "FF BASED MCAP",
        "ORD SHARES",
        "ORD SHARES MCAP",
        "VOLUME",
    ]
    print(df[numeric_cols].describe().round(2))


def sector_distribution(df):
    print("\n" + "=" * 60)
    print("SECTOR/INDUSTRY DISTRIBUTION")
    print("=" * 60)
    latest = df[df["Date"] == df["Date"].max()].copy()
    print(f"Number of constituents: {len(latest)}")
    print(f"\nIndex weight distribution:")
    print(
        latest[["SYMBOL", "IDX WT %"]].sort_values("IDX WT %", ascending=False).head(10)
    )
    print(f"\nTop 10 by market cap (FF BASED MCAP):")
    print(
        latest[["SYMBOL", "FF BASED MCAP"]]
        .sort_values("FF BASED MCAP", ascending=False)
        .head(10)
    )


def price_statistics(df):
    print("\n" + "=" * 60)
    print("PRICE STATISTICS BY SYMBOL")
    print("=" * 60)
    price_stats = (
        df.groupby("SYMBOL")["PRICE"].agg(["mean", "min", "max", "std"]).round(2)
    )
    price_stats["cv"] = (price_stats["std"] / price_stats["mean"] * 100).round(2)
    print(price_stats.sort_values("mean", ascending=False).head(15))


def volume_statistics(df):
    print("\n" + "=" * 60)
    print("VOLUME STATISTICS BY SYMBOL")
    print("=" * 60)
    vol_stats = (
        df.groupby("SYMBOL")["VOLUME"].agg(["mean", "min", "max", "std"]).round(0)
    )
    print(vol_stats.sort_values("mean", ascending=False).head(15))


def time_series_overview(df):
    print("\n" + "=" * 60)
    print("TIME SERIES OVERVIEW")
    print("=" * 60)
    daily_agg = (
        df.groupby("Date")
        .agg(
            {
                "PRICE": "mean",
                "VOLUME": "sum",
                "FF BASED MCAP": "sum",
                "IDX WT %": "sum",
            }
        )
        .reset_index()
    )
    print(f"Trading days: {len(daily_agg)}")
    print(f"\nIndex stats:")
    print(daily_agg.describe())


def plot_price_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    latest = df[df["Date"] == df["Date"].max()]
    ax.hist(latest["PRICE"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Price (PKR)")
    ax.set_ylabel("Frequency")
    ax.set_title("KSE-30 Price Distribution (Latest Date)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
    plt.close()


def plot_index_weights(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    latest = df[df["Date"] == df["Date"].max()].sort_values("IDX WT %", ascending=True)
    ax.barh(latest["SYMBOL"], latest["IDX WT %"])
    ax.set_xlabel("Index Weight (%)")
    ax.set_title("KSE-30 Constituent Weights (Latest Date)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "index_weights.png", dpi=150)
    plt.close()


def plot_volume_time_series(df):
    daily_vol = df.groupby("Date")["VOLUME"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_vol["Date"], daily_vol["VOLUME"] / 1e6, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (Millions)")
    ax.set_title("KSE-30 Total Daily Volume")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "volume_timeseries.png", dpi=150)
    plt.close()


def plot_mcap_time_series(df):
    daily_mcap = df.groupby("Date")["FF BASED MCAP"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_mcap["Date"], daily_mcap["FF BASED MCAP"] / 1e12, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Free Float MCAP (Trillions PKR)")
    ax.set_title("KSE-30 Total Free Float Market Cap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcap_timeseries.png", dpi=150)
    plt.close()


def plot_correlation_matrix(df):
    numeric_cols = ["PRICE", "IDX WT %", "FF BASED SHARES", "FF BASED MCAP", "VOLUME"]
    latest = df[df["Date"] == df["Date"].max()][numeric_cols]
    corr = latest.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix (Latest Date)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def main():
    print("Loading data...")
    df = load_data()

    basic_info(df)
    missing_values(df)
    numeric_summary(df)
    sector_distribution(df)
    price_statistics(df)
    volume_statistics(df)
    time_series_overview(df)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS...")
    print("=" * 60)

    plot_price_distribution(df)
    plot_index_weights(df)
    plot_volume_time_series(df)
    plot_mcap_time_series(df)
    plot_correlation_matrix(df)

    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
