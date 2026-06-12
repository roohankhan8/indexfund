# Production Pipeline

This folder is the canonical executable project.

## Run Everything

```bash
python run_all.py
```

## Run One Stage

```bash
python run_all.py --stage pipeline
```

## Stage Order

1. `prepare`
2. `pipeline`
3. `eda_raw`
4. `eda_master`
5. `stationarity`
6. `chapter3`
7. `report`
8. `risk_map`

## Inputs

Runtime inputs live in `data/raw/`.

## Outputs

- `output/analysis/` contains the main pipeline outputs
- `output/eda_raw/` contains raw-panel EDA outputs
- `output/eda_master/` contains master-dataset EDA outputs

Report assets are written into `../docs/report_workspace/`.
