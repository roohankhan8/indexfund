# Equity Market Volatility Using GARCH Models: Evidence from Pakistan Stock Exchange (KSE-100 Index) (2016)

## Paper Details
- Authors: Muhammad Asif, Abdul Aziz
- Journal: International Journal of Accounting and Economics Studies
- Volume/Issue: 4(2)
- Year: 2016
- DOI: 10.14419/ijaes.v4i2.6200
- Market focus: Pakistan Stock Exchange benchmark index (KSE-100)

## Objective
The paper examines volatility clustering in KSE-100 returns and evaluates which ARCH-family specification best captures PSX return dynamics, while also discussing implications for weak-form market efficiency.

## Methodology Overview
1. Collect daily KSE-100 index observations.
2. Compute return series and test distributional and stationarity properties.
3. Estimate multiple ARCH-family models:
- GARCH
- EGARCH
- PGARCH
- TARCH
4. Select model based on information criteria and fit quality.
5. Interpret persistence, clustering, and efficiency implications.

## Model Used
- Core model family: ARCH/GARCH variants
- Candidate models explicitly discussed: GARCH(1,1), EGARCH, PGARCH, TARCH
- Model-selection criteria: AIC, SIC, and Log-Likelihood

## Data Used
- Dataset: Daily KSE-100 index series
- Period: 1 January 2008 to 31 December 2015
- Sample size: 1983 observations
- Variable focus: index returns/volatility behavior

## Where the Data Came From
- Source stated in paper: KSE website (PSX historical index data)

## Key Findings
- Best-performing model among tested ARCH-family variants: GARCH(1,1).
- Return series exhibits volatility clustering and leptokurtic behavior (fat tails).
- Authors report weak-form efficiency interpretation while also noting random-walk rejection from model/nonparametric evidence in their result discussion; this indicates interpretive tension in wording, but the volatility-dynamics result is clear.

## Limitations
- The paper explicitly notes limitation around re-composition of KSE-100 constituents over time.
- Single-index focus (KSE-100) limits cross-index generalization.
- Daily frequency captures short-run dynamics but not necessarily structural macro regime shifts.
- ARCH-family models rely on parametric assumptions and may under-handle extreme discontinuities.

## Research Gap
The study addresses a local evidence gap on PSX volatility modeling by comparatively testing ARCH-family variants for KSE-100 over an extended post-crisis period, with practical emphasis on identifying a robust volatility model for Pakistani equities.

## Practical Relevance
For PSX risk and trading workflows, the paper supports using GARCH(1,1) as a practical baseline model for day-to-day volatility estimation, risk control, and portfolio decision support under clustered-volatility conditions.
