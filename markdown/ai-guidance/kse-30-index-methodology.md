# KSE-30 Index Methodology (PSX) - Coding Guidance

Source PDF: `0-docs/KSE-30_Index_updated_version.pdf`

## What This Document Is For
Use this as the canonical rule reference when implementing KSE-30 recomposition logic, eligibility checks, ranking, and index maintenance workflows.

## Core Methodology
- KSE-30 is a free-float market capitalization index with 30 constituents.
- Base period/value: June 2005 = 10,000.
- Index level is based on free-float market value of constituents relative to divisor/base.

## Eligibility Pre-Requisites (Screening)
A company is not eligible if any of these fail:
- Not on Defaulters' Segment.
- Not suspended / non-tradable in preceding 6 months from recomposition date.
- Security available in CDC.
- At least 2 months formal listing history on PSX.
- At least 1 financial year operational track record.
- Minimum free-float shares: 5% of total outstanding shares.
- Traded on at least 75% of total trading days.
- Average Impact Cost <= 1.5%.
- Open-end and closed-end mutual funds are excluded.

## Selection and Ranking
- Final rank combines:
- 50% weight: free-float market capitalization (higher is better).
- 50% weight: liquidity via Impact Cost (lower is better).
- Top 30 by final rank are selected.

## Recomposition Schedule
Semi-annual recomposition cycle:
- As of June 30 -> revision effective around September 15.
- As of December 31 -> revision effective around March 15.

Implementation note:
- Dates should be handled as configurable calendar events (do not hardcode weekends/holidays).
- Effective implementation may shift to next trading day by exchange notice.

## Corporate Action Handling
- Cash dividend: no adjustment (unlike KSE-100 treatment).
- Bonus/right/new capital actions require divisor adjustments to preserve index continuity.
- Ex-price and divisor timing follow PSX rulebook references and book closure mechanics.

## Data Model Recommendations
Represent at minimum:
- `symbol`
- `as_of_date`
- `free_float_shares`
- `free_float_factor` (if using banding logic)
- `closing_price`
- `free_float_market_cap`
- `impact_cost_avg_6m`
- `trading_days_ratio`
- `eligibility_flags` (default/suspension/cdc/listing_age/operational_history/free_float_min/impact_cost_limit)
- `final_score`
- `rank`
- `is_constituent`

## Computation Pipeline (Suggested)
1. Ingest as-of universe and corporate actions.
2. Apply eligibility filters.
3. Compute free-float market cap.
4. Compute liquidity metric (impact cost).
5. Normalize/score both legs and aggregate final score.
6. Rank and pick top 30.
7. Produce `incoming` and `outgoing` vs previous composition.
8. Attach implementation-effective date from official PSX notice.

## Validation Checks
- Constituents count must equal 30.
- No ineligible symbol appears in final basket.
- Incoming/outgoing sets reconcile with previous and new basket.
- Re-run with same inputs must be deterministic.

## Practical Caveat
This PDF text extraction contains some encoding artifacts in bullets. If any clause seems ambiguous, verify against the original PDF wording before finalizing production rules.
