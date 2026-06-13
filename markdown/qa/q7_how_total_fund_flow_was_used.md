# Q7: How `total_fund_flow` was used in the project, what results it generated, and how it was helpful

This file explains the role of `total_fund_flow` after it is calculated.

`total_fund_flow` is one of the main monthly target variables in the final pipeline. It is used for:

- exploratory analysis
- correlation and macro-flow analysis
- stationarity and Granger-causality testing
- out-of-sample forecasting with Naive, ARIMAX, and VAR models
- final reporting and dashboard summaries
- practical portfolio-tilt interpretation

## 1. Where it is used in the pipeline

### A. As a core monthly feature in the master dataset

After construction, `total_fund_flow` is saved into the monthly master table along with related columns such as:

- `composite_net_flow`
- `flow_pct_sector`
- `flow_spike_sector`

You can see these columns in [production_pipeline/output/analysis/monthly_master.csv](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/output/analysis/monthly_master.csv:1).

The pipeline explicitly keeps these columns when finalizing the monthly dataset in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:499).

### B. For exploratory fund-flow visualization

The project plots monthly inflow and outflow bars using `total_fund_flow` in the EDA section. Positive months are shown in blue and negative months in red, with spike months highlighted.

This happens in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:561), which generates:

- `E03_fund_flows.png`

This figure is used to show the time pattern of aggregate inflows and outflows before forecasting.

### C. For correlation analysis

The normalized version, `flow_pct_sector`, is included in the monthly correlation heatmap with macro and index variables.

This happens in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:595).

This helps compare flow behavior against:

- oil returns
- USD/PKR returns
- interest rate
- CPI
- KSE-30 return
- KSE-30 volatility

### D. As the main forecasting target

In the final fund-flow prediction section, the pipeline sets:

```text
TARGET = "total_fund_flow"
```

That is done in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:791) to [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:793).

So the project is not forecasting price directly in this section. It is forecasting the monthly net flow proxy itself.

## 2. What analysis it enabled

### A. Stationarity testing

The pipeline runs an ADF-style stationarity test on `total_fund_flow` before modeling.

This is part of [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:799).

This matters because the project needed to check whether the flow series was statistically suitable for time-series modeling.

### B. Granger-causality testing

The pipeline tests whether lagged macro variables help predict `total_fund_flow`.

This is done in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:820).

The tested relationships are:

- interest rate -> flow
- CPI -> flow
- oil return -> flow
- USD/PKR return -> flow

The project then plots the resulting p-values in:

- `FF02_granger.png`

saved from [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:936).

### C. Out-of-sample forecasting

`total_fund_flow` is the target used for the three final forecast comparisons:

- Naive random-walk benchmark
- `ARIMAX(1,0,1)`
- `VAR(1)`

These are implemented in:

- [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:857) for ARIMAX
- [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:892) for VAR

The project also generates the main prediction chart:

- `FF01_total_flow_predictions.png`

from [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:915).

## 3. What results it generated

The final model-comparison table is saved in:

- [production_pipeline/output/analysis/results_fund_flow.csv](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/output/analysis/results_fund_flow.csv:1)

Those final stored results are:

| Model | RMSE | MAE | R2 | Directional Accuracy |
|---|---:|---:|---:|---:|
| Naive (RW) | 83.35 | 51.36 | -1.4196 | 37.5% |
| ARIMAX(1,0,1) | 58.00 | 36.52 | -0.1716 | 70.8% |
| VAR(1) | 62.54 | 38.80 | -0.3622 | 75.0% |

These rows are written by [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:950).

## 4. What those results mean

### A. The series was useful, but mainly directionally

The project's own interpretation is that `total_fund_flow` is hard to predict precisely in PKR magnitude, but still contains usable dynamic information.

That interpretation is stated in [markdown/report-workspace/current-FYP Report (Analyzing Mutual Funds).md](/abs/path/D:/roohan/fyp-indexfunds/markdown/report-workspace/current-FYP Report (Analyzing Mutual Funds).md:791).

In practical terms:

- exact point forecasts were still noisy
- out-of-sample `R2` stayed negative
- but ARIMAX and VAR clearly beat the naive benchmark on direction

### B. Directional accuracy was the key operational result

The main practical gain was that the better models improved inflow/outflow direction prediction substantially versus the naive benchmark.

From the final stored results:

- Naive: `37.5%`
- ARIMAX: `70.8%`
- VAR: `75.0%`

This is why the project treats the signal as a decision-support variable, not a precise money-amount estimator.

The report explanation says the models are better interpreted as directional tools rather than exact point estimators in [chapter-05-explanations.md](/abs/path/D:/roohan/fyp-indexfunds/markdown/report-workspace/graph_explanations/chapter-05-explanations.md:44).

## 5. How it was helpful in the project

### A. It turned raw fund data into a measurable investor-behavior signal

Without `total_fund_flow`, the project would only have:

- stock/index price information
- macro variables
- raw fund NAV/AUM series

`total_fund_flow` converts the fund data into a cleaner monthly signal representing net capital movement into or out of the tracked KSE-30-related fund group.

That made investor activity usable as a model target.

### B. It gave the project a central monthly target for the fund-flow chapter

The whole fund-flow modeling section is built around this series:

- stationarity testing
- Granger-causality testing
- forecast benchmarking
- actual-vs-predicted visualization

Without this variable, Section 5 of the production pipeline would not have a concrete monthly flow target to model.

### C. It supported a regime-style interpretation for portfolio use

The report explicitly connects directional fund-flow signals to practical allocation decisions.

In [docs/reports/root_exports/_docx_inspect/document_paragraphs.txt](/abs/path/D:/roohan/fyp-indexfunds/docs/reports/root_exports/_docx_inspect/document_paragraphs.txt:789), the report says the ARIMAX-generated directional flow signals can act as a macro-flow regime filter:

- predicted inflows -> tilt toward high-beta, liquid core constituents
- predicted outflows -> shift more defensively

This is one of the clearest examples of how `total_fund_flow` helped beyond pure analysis.

### D. It linked the fund-flow study to the rebalancing application

The project is not only descriptive. It tries to move from explanation to application.

The report recommends using directional flow signals together with predictive rebalancing models as a decision-support tool for portfolio tilts ahead of official index reviews in [markdown/report-workspace/current-FYP Report (Analyzing Mutual Funds).md](/abs/path/D:/roohan/fyp-indexfunds/markdown/report-workspace/current-FYP Report (Analyzing Mutual Funds).md:863).

So `total_fund_flow` helped bridge:

- market analysis
- forecasting
- portfolio action

## 6. Where it appears in final outputs

`total_fund_flow` directly contributes to these outputs:

- [production_pipeline/output/analysis/monthly_master.csv](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/output/analysis/monthly_master.csv:1)
- [production_pipeline/output/analysis/results_fund_flow.csv](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/output/analysis/results_fund_flow.csv:1)
- `production_pipeline/output/analysis/figures/eda/E03_fund_flows.png`
- `production_pipeline/output/analysis/figures/fund_flow/FF01_total_flow_predictions.png`
- `production_pipeline/output/analysis/figures/fund_flow/FF02_granger.png`
- `production_pipeline/output/analysis/figures/summary/SUMMARY_dashboard.png`

It is also used in the summary dashboard:

- as the directional-accuracy scorecard input in Panel 2
- as the total-flow chart in Panel 4

That dashboard logic is in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:1472) and [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:1500).

## 7. Balanced conclusion

`total_fund_flow` was helpful because it gave the project a usable proxy for investor capital movement in KSE-30-related funds.

Its biggest contribution was not perfect numeric forecasting. Its value was that it:

- made monthly fund behavior measurable
- enabled macro-versus-flow testing
- produced a forecast target for ARIMAX and VAR
- generated materially better inflow/outflow direction signals than the naive benchmark
- supported the project's practical portfolio-tilt interpretation

So in this project, `total_fund_flow` is best understood as a **useful directional regime signal built from fund data**, not as a perfectly predictable cash-flow amount.
