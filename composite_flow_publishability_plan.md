# Composite Flow Publishability Plan

## Objective

Make the project publishable with minimal disruption by replacing the implied market-wide "KSE-30 net flow" target with a clearly defined 3-fund composite net-flow series built from AKD, NBP, and NTI.

## Why this is the minimal-change fix

- It preserves the current data sources.
- It preserves the current forecasting models: naive benchmark, ARIMAX-style model, and VAR.
- It preserves downstream evaluation, plots, and most existing variable names.
- It removes the main methodological weakness raised by the examiner: unsupported use of an unofficial aggregate flow measure.

## New target definition

For fund `i` in month `t`:

`DollarNetFlow_(i,t) = TNA_(i,t) - TNA_(i,t-1) * (1 + R_(i,t))`

Define lagged composite weights:

`w_(i,t-1) = TNA_(i,t-1) / sum_j TNA_(j,t-1)`

Define composite return:

`CompositeReturn_t = sum_i w_(i,t-1) * R_(i,t)`

Define composite assets:

`CompositeTNA_t = sum_i TNA_(i,t)`

Define composite net flow:

`CompositeNetFlow_t = CompositeTNA_t - CompositeTNA_(t-1) * (1 + CompositeReturn_t)`

Define normalized composite flow:

`CompositeFlowPct_t = CompositeNetFlow_t / CompositeTNA_(t-1)`

## Implementation strategy

### 1. Keep compatibility

To minimize code breakage:

- keep `total_fund_flow` as the main target column used by downstream models
- change its internal construction so it now equals `CompositeNetFlow_t`
- keep `flow_pct_sector` as the normalized version of the composite flow

### 2. Add transparent intermediate columns

Add these fields to the monthly master where possible:

- `composite_return_monthly`
- `composite_net_flow`
- `sector_aum_mn`
- `sector_aum_prev`

Then set:

- `total_fund_flow = composite_net_flow`

### 3. Keep fund-level flows

Do not remove:

- `flow_akd`
- `flow_nbp`
- `flow_nti`

They remain useful descriptive variables and let the report discuss fund-specific flow behavior, but they should no longer be directly summed and treated as the core target without composite-return adjustment.

## Report wording changes

Replace:

- "KSE-30 fund flow"
- "aggregate KSE-30 net flow"
- "official KSE-30 flow"

With:

- "3-fund composite net flow"
- "proxy KSE-30-related composite flow"
- "sampled sector-flow series constructed from AKD, NBP, and NTI"

## Required code changes

### Pipeline files

- `5_claude_pipeline/pipeline.py`
- `6_cursor_model/pipeline.py`
- `7_codex_model/pipeline.py`

### Documentation/report files

- `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt`
- `report_workspace_2/chapter-05-results-and-analysis/chapter-05-results-and-analysis.txt`
- `presentation_concepts_and_theories_explainer.md`
- `overview/README.md`
- `equations_used_and_sources.md`

## Validation after patching

Check that:

- `total_fund_flow` is now built from composite TNA and composite return
- ARIMAX and VAR still run without interface changes
- figure titles say "composite" or "proxy sector-flow" rather than implying official market-wide flow
- methodology text states that NAV-based return is a proxy if total return is unavailable

## Publishable positioning

Recommended one-sentence framing:

"This study constructs an AUM-weighted composite index of three KSE-30-related funds and estimates the net flow of that composite using the standard return-adjusted total-net-assets approach."
