# Fund Flow Methodology Fix

## What needed correction

The examiner's concern is valid in two places:

1. The project used a fund-flow equation without citing the mutual-fund flow literature.
2. The report wording risked presenting a constructed sample-based flow series as if it were an official observed KSE-30 net-flow measure.

## Correct methodological position

The academically safer statement is:

`DollarNetFlow_(i,t) = TNA_(i,t) - TNA_(i,t-1) * (1 + R_(i,t))`

where:

- `TNA` = total net assets or AUM
- `R_(i,t)` = period total return of fund `i`

This is the standard return-adjusted asset-growth logic used to infer investor net flow from mutual-fund asset data.

## How this project should describe its implementation

Because direct total-return series were not separately available for all sampled funds, the project used a NAV-based proxy:

`flow_t = AUM_t - AUM_(t-1) * (NAV_t / NAV_(t-1))`

This should be described as an approximation to the literature-standard flow measure, not as the strongest possible version of the equation.

## How `total_fund_flow` should be described

The project combines AKD, NBP, and NTI:

`total_fund_flow_t = flow_AKD,t + flow_NBP,t + flow_NTI,t`

This should be described as:

- a proxy aggregate sector-flow series
- a sampled KSE-30-related mutual-fund flow measure

It should not be described as:

- an official KSE-30 net-flow series
- the market-wide observed net flow of the KSE-30

## Suggested report wording

"Following the mutual-fund flow literature, investor net flow is proxied by the return-adjusted change in total net assets. At the fund level, net flow is defined as `TNA_(i,t) - TNA_(i,t-1)(1 + R_(i,t))`. Because a direct total-return series was not separately available for all sampled funds, the empirical implementation uses monthly NAV-based return as a practical proxy for `R_(i,t)`. The aggregate series used in this study is therefore not an official observed KSE-30 net-flow measure, but a proxy sector-flow series constructed from AKD, NBP, and NTI."

## Core literature to cite

- Sirri, E. R., and Tufano, P. (1998). *Costly Search and Mutual Fund Flows*.
- Edelen, R. M. (1999). *Investor Flows and the Assessed Performance of Open-End Mutual Funds*.
- Berk, J. B., and Green, R. C. (2004). *Mutual Fund Flows and Performance in Rational Markets*.

## Files already patched

- `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt`
- `report_workspace_2/chapter-05-results-and-analysis/chapter-05-results-and-analysis.txt`
- `presentation_concepts_and_theories_explainer.md`
- `overview/README.md`
- `equations_used_and_sources.md`
