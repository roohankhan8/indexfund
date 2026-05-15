# Determining the Efficacy of GARCH Type Models for Estimating VaR in Case of Equities Enlisted in PSX (2024)

## Paper Details
- Type: Undergraduate Final Year Project Report
- Institution: NED University of Engineering and Technology (Department of Mathematics, BSCF)
- Completion date noted in report: August 2024
- Study domain: Risk measurement for PSX equities using VaR and GARCH volatility modeling

## Objective
The report examines whether dynamic-volatility VaR (via GARCH-type volatility estimates) is more effective than static-variance VaR for PSX equity risk measurement and decision support.

## Methodology Overview
1. Construct a hypothetical equity portfolio from KSE-30 constituents.
2. Use daily data over a multi-year sample window.
3. Estimate volatility through multiple GARCH(p,q) specifications.
4. Select best-fitting models using model diagnostics (AIC, log-likelihood, persistence behavior).
5. Compute VaR using a variance-covariance framework under:
- Time-invariant (static) variance
- Time-varying (dynamic) conditional variance from GARCH
6. Compare model outputs against realized losses (breach-based and distance-based checks).

## Model Used
### Risk Model
- Value at Risk (VaR), mainly variance-covariance implementation in this empirical section.

### Volatility Models
- Multiple GARCH(p,q) variants were tested across portfolio constituents and market-value return series.
- The report identifies GARCH(2,3) and GARCH(1,2) as the most appropriate models in its setting.

### Model Selection Indicators
- Akaike Information Criterion (AIC)
- Log-likelihood
- Volatility persistence interpretation through alpha/beta terms

## Data Used
- Market universe: KSE-30 linked equities (PSX)
- Frequency: Daily prices/returns
- Study period: January 2019 to December 2023
- Portfolio setup: Hypothetical PKR 1 billion portfolio
- Portfolio rule noted: rebalance/review when a security's market value drops below 10% of original investment cost
- Decision signal referenced: Price-to-Earnings (P/E) ratio for position adjustment logic

## Where the Data Came From
- Primary source explicitly stated in methodology: PSX website
- Dataset includes daily share-price based observations for KSE-30 constituents over the analysis window

## Key Findings
- Dynamic-variance VaR (GARCH-based) is reported as more effective than static-variance VaR.
- Two practical reasons highlighted by the report:
- It is closer to historically observed losses.
- It generates more VaR-limit breaches/action points for portfolio managers.
- Best volatility-model choices in this work were GARCH(2,3) and GARCH(1,2), depending on series.

## Interpretation of Reported Comparative Outputs
The report compares time-invariant VaR and GARCH-based VaR using distance and breach counts by company. It concludes that, overall, the GARCH-driven approach better captures time-varying volatility and improves VaR usefulness for ongoing risk monitoring.

## Limitations
- The analysis is constrained to PSX/KSE-30 scope and one national market context.
- VaR framework assumptions (especially normality in variance-covariance settings and regular market conditions) remain structural constraints.
- Model performance can be portfolio-specific and period-specific; findings may not transfer uniformly across regimes.
- Backtesting design is focused on historical comparison and may not fully capture rare external shocks.
- The report itself notes broader market-structure frictions in Pakistan (transparency, sentiment-led behavior), which can affect model reliability in practice.

## Research Gap
The report explicitly positions a local practice gap: VaR is commonly calculated for reporting/control in Pakistan, but less used by risk-takers for decision-making and less explored through alternative dynamic-volatility approaches in PSX context. It addresses this by testing GARCH-based dynamic variance within a practical portfolio workflow.

## Practical Relevance
For PSX risk workflows, this report supports replacing or complementing static-variance VaR with dynamic conditional-variance estimates (GARCH family), especially where volatility clustering is material and actionable risk signals are needed for portfolio adjustments.
