# Q6: How `total_fund_flow` is calculated in this project

`total_fund_flow` is not imported from any external source. The project derives it from the monthly NAV and AUM behavior of three tracked funds: `AKD`, `NBP`, and `NTI`.

## 1. Fund-level monthly flow

For each fund, the pipeline first converts daily data into monthly data by taking:

- `nav_start`: first NAV in the month
- `nav_end`: last NAV in the month
- `aum`: last AUM in the month

This happens in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:328).

Then it computes:

```text
aum_prev     = previous month's AUM
nav_return_m = (nav_end / nav_start) - 1
fund_flow    = aum - aum_prev * (1 + nav_return_m)
fund_flow_pct = fund_flow / aum_prev
```

This is implemented in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:335) to [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:340), and the same logic also appears in [production_pipeline/eda_kse30.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/eda_kse30.py:113).

Meaning:

- If AUM increased only because the fund's NAV went up, `fund_flow` stays near zero.
- If AUM increased more than NAV performance alone would explain, the extra part is treated as investor inflow.
- If AUM fell more than NAV performance alone would explain, it is treated as investor outflow.

So the project is using a return-adjusted AUM-change formula, not a raw month-to-month AUM difference.

## 2. Data repair before flow calculation

Before calculating monthly fund flow, the pipeline repairs isolated zero-AUM months when NAV still exists. Those zeros are treated as data gaps, then internally interpolated.

That logic is in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:76).

This matters because otherwise a bad zero would create a fake liquidation spike and distort fund flow.

## 3. Sector-level / total flow construction

After monthly fund-level values are prepared for `AKD`, `NBP`, and `NTI`, the pipeline merges them into one monthly table and builds the aggregate series.

First it computes combined sector AUM:

```text
sector_aum_mn = aum_akd + aum_nbp + aum_nti
sector_aum_prev = previous month's sector_aum_mn
```

Then it computes previous-month AUM weights:

```text
w_akd_prev = aum_akd(t-1) / sector_aum_prev
w_nbp_prev = aum_nbp(t-1) / sector_aum_prev
w_nti_prev = aum_nti(t-1) / sector_aum_prev
```

Then it builds a composite monthly return:

```text
composite_return_monthly =
    w_akd_prev * nav_return_akd_monthly
  + w_nbp_prev * nav_return_nbp_monthly
  + w_nti_prev * nav_return_nti_monthly
```

Finally it computes:

```text
composite_net_flow =
    sector_aum_mn - sector_aum_prev * (1 + composite_return_monthly)

total_fund_flow = composite_net_flow
```

This is implemented in [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:459) to [production_pipeline/pipeline.py](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/pipeline.py:480).

## 4. What `total_fund_flow` really represents

In this repo, `total_fund_flow` is the **3-fund composite net flow proxy** for the tracked KSE-30-related fund set, not an official published market-wide KSE-30 flow series.

That design choice is stated in [PROJECT_REFERENCE.md](/abs/path/D:/roohan/fyp-indexfunds/PROJECT_REFERENCE.md:17).

## 5. Simple interpretation

- Positive `total_fund_flow`: net money entered the 3-fund group that month.
- Negative `total_fund_flow`: net money left the 3-fund group that month.
- Zero or near-zero `total_fund_flow`: AUM changes were mostly explained by NAV performance rather than subscriptions/redemptions.

## 6. Where the final values are stored

The final monthly series is written to:

- [production_pipeline/output/analysis/monthly_master.csv](/abs/path/D:/roohan/fyp-indexfunds/production_pipeline/output/analysis/monthly_master.csv:1)

Relevant columns there are:

- `composite_net_flow`
- `total_fund_flow`
- `flow_pct_sector`
- `flow_spike_sector`

At the moment, `total_fund_flow` and `composite_net_flow` are the same column under two names.
