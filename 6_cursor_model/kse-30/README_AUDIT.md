# KSE-30 Recomposition Audit (6_cursor_model)

Reference: `.ai-guidance/kse-30-index-methodology.md`
Audited pipeline: `6_cursor_model/pipeline.py`

## Result
Current Section 7 in `pipeline.py` is **not PSX-methodology compliant** for recomposition.

## Key Gaps Found
1. Recomposition trigger mismatch
- Current logic uses any day-to-day constituent set change as a rebalance event.
- KSE-30 rule requires semi-annual review basis (June 30, December 31) with effective implementation by notice.

2. Missing eligibility gate
- No explicit screening for prerequisites (free-float >= 5%, trading-days ratio >= 75%, listing age >= 2 months, etc.).

3. Missing 50/50 final ranking
- No final rank computed as 50% free-float market cap + 50% liquidity score (impact cost leg).

4. No deterministic top-30 selection from rules
- Current output is predictive ML scoring of retention/weights, not methodology-defined recomposition selection.

5. No explicit incoming/outgoing reconciliation table
- Required rule-based incoming/outgoing vs prior basket not generated as a methodology artifact.

## What Was Added
A new rule-based recomposition pipeline was added:
- `6_cursor_model/kse-30/recomposition_pipeline_kse30.py`

This script implements:
- Semi-annual as-of schedule (June 30 / December 31; snap to nearest trading day)
- Eligibility checks available from dataset
- 50/50 final score using:
  - free-float market cap score
  - liquidity proxy score (volume-based proxy because impact-cost data is unavailable in current dataset)
- Top-30 constituent selection
- Incoming/outgoing event tables
- Deterministic CSV outputs in `6_cursor_model/kse-30/`

## Data Limitation
Current source data does not include direct PSX impact-cost series, defaulters segment flags, CDC status, or explicit suspension flags. The new script marks these as unavailable and uses a liquidity proxy for the ranking leg.
