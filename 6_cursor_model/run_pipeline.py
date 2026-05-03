"""Thin entry point for `pipeline.py` in this folder (same behaviour).

Usage::

    python 6_cursor_model/run_pipeline.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent / "pipeline.py"
    runpy.run_path(str(here), run_name="__main__")


if __name__ == "__main__":
    main()
