# Q2: Difference between `6_cursor_model/pipeline.py` and `6_cursor_model/run_pipeline.py`

## Short answer
`run_pipeline.py` does **not** implement a separate pipeline. It simply executes `pipeline.py`.

## Detailed difference
1. **`pipeline.py`**
- Full implementation file (~88 KB).
- Contains all sections: data loading, cleaning, feature engineering, modeling, evaluation, plots, CSV outputs, summary.
- This is where all logic actually lives.

2. **`run_pipeline.py`**
- Tiny wrapper (~374 bytes).
- Uses `runpy.run_path()` to run `pipeline.py` as `__main__`.
- Adds no modeling logic, no data changes, and no output differences by itself.

## Behavior/output impact
- Running either command should produce the same outputs:
  - `python 6_cursor_model/pipeline.py`
  - `python 6_cursor_model/run_pipeline.py`

## Why keep both files?
- `run_pipeline.py` is a cleaner entry point (useful for tooling/scripts).
- `pipeline.py` remains the main source file for code edits and debugging.
