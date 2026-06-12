from __future__ import annotations

import argparse
import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

STAGES = {
    "prepare": "prepare_kse30_basic.py",
    "pipeline": "pipeline.py",
    "eda_raw": "eda_kse30.py",
    "eda_master": "eda_master_processed.py",
    "stationarity": "transform_for_stationarity.py",
    "chapter3": "generate_ch3_eda_graphs.py",
    "report": "generate_report_assets.py",
    "risk_map": "generate_rebalancing_risk_map_with_labels.py",
}


def run_stage(stage: str) -> None:
    script = BASE_DIR / STAGES[stage]
    print(f"\n=== Running {stage}: {script.name} ===")
    runpy.run_path(str(script), run_name="__main__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full KSE-30 production pipeline from input preparation to report assets."
    )
    parser.add_argument(
        "--stage",
        choices=["all", *STAGES.keys()],
        default="all",
        help="Run a single stage or the full production workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ordered_stages = [
        "prepare",
        "pipeline",
        "eda_raw",
        "eda_master",
        "stationarity",
        "chapter3",
        "report",
        "risk_map",
    ]
    stages = ordered_stages if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage)


if __name__ == "__main__":
    main()
