# FYP Index Funds - Project Overview

**Analyzing KSE-30 (Pakistan's Top 30 Index) Using Quantitative Finance**

*Note: KSE-30 is a stock market index (top 30 companies). Mutual funds (AKD, NBP, NTI) track this index. We analyze KSE-30 directly and use mutual fund data as external variables - since flows in index-tracking mutual funds reflect flows in KSE-30.*

---

## Project Goal

This project builds an automated system to analyze and predict the KSE-30 index behavior in Pakistan's stock market. We use mutual fund flows (AKD, NBP, NTI) as external indicators - since these funds track KSE-30, their flows reflect KSE-30 flows.

We answer questions like:
- Will investors put more money in or take it out?
- How volatile is the market?
- How should the fund portfolio be adjusted (rebalanced)?
- Is the market efficient?

---

## Folder Structure & What Each Model Does

### Folder 0: Data Extraction & Documentation
**Purpose:** Gather raw data from Pakistan Stock Exchange (PSX)

| Component | What It Does |
|-----------|--------------|
| `0a_data_extraction/` | Downloads daily KSE-30 stock prices from PSX |
| `0b-raw-data/` | Stores raw Excel files, zips, and source data |
| `0-docs/` | Project proposal, research papers, final reports |

**Output:** Raw stock prices, fund NAVs, macro economic data (2018-2026)

---

### Folder 1: Data Cleaning & EDA
**Purpose:** Clean data and explore patterns through visualization

| Component | What It Does |
|-----------|--------------|
| `1a_data_cleaning/` | Removes duplicates, fixes missing values, standardizes symbols |
| `1b_eda/` | Exploratory data analysis only - generates visualizations and explores patterns |

**What 1b_eda Contains:**
- EDA scripts (eda_kse30.py, eda_master_processed.py)
- EDA visualizations in output folders
- Stationarity tests (ADF) to determine if data is suitable for modeling

**Note:** The `daily_master.csv` and `monthly_master.csv` were generated later in Folder 4/5 (Claude pipeline).

---

### Folder 4: Claude Model (Initial Models)
**Purpose:** Build individual models for each financial analysis task

| Model | File | What It Does | Why It Matters |
|-------|------|--------------|----------------|
| **Preprocessing** | nb0_preprocessing.py | Cleans data, creates daily/monthly master tables | Foundation for all analysis |
| **EDA** | nb1_eda.py | Explores data patterns visually | Understands fund behavior |
| **Fund Flow Prediction** | nb7_kse30_fund_flow_prediction.py | Predicts 3-fund composite KSE-30 tracker flow using ARIMAX + VAR models | Uses mutual fund data from AKD, NBP, and NTI to build an AUM-weighted composite-flow proxy |
| **Volatility Modeling** | nb3_garch_volatility.py | Models market volatility using GARCH | Captures risk clustering |
| **Portfolio Optimization** | nb4_portfolio_optimisation.py | Mean-variance optimization (Markowitz) | Shows optimal stock weights |
| **Rebalancing Prediction** | nb4b_rebalancing_prediction.py | Predicts rebalancing using Random Forest (Regressor + Classifier) | Key for index tracking |
| **Market Efficiency** | nb5_market_efficiency.py | Tests if prices follow random walk (variance ratio, Granger causality) | Validates predictability |
| **Summary** | nb6_results_summary.py | Aggregates all results | Final report generation |

---

### Folder 5: Claude Pipeline (Unified System)
**Purpose:** Combine all models into one automated pipeline

**What it does:**
- Runs all analyses sequentially in one command
- Produces standardized output files
- Creates the master datasets used by all subsequent models

**Output:** Consolidated results ready for analysis

---

### Folder 6: Cursor Model (Improved)
**Purpose:** Fix methodology issues and improve accuracy

**Improvements:**
- Better stationarity handling (use log returns instead of levels)
- Enhanced rebalancing methodology
- Added cross-validation to prevent overfitting
- More robust prediction metrics

---

### Folder 7: Codex Model (Refinements)
**Purpose:** Fine-tune models based on results

**What was done:**
- Refined rebalancing forecasts
- Improved prediction accuracy
- Generated cleaner CSV outputs

---

### Folder 8: Last Model (Final Version)
**Purpose:** Final production-ready model

**What was done:**
- KSE-30 specific rebalancing pipeline
- Optimized for real-world use
- Clean output generation

---

### Folder 9: Backtesting (March 2026)
**Purpose:** Validate the model on unseen data

**What it does:**
- Ran the pipeline on March 2026 data (held out from training)
- Compared predictions vs actual outcomes
- Generated performance metrics

**Output:**
- `march_2026_metrics.json` - How accurate were predictions?
- `march_2026_rebalance_actual_vs_pred.csv` - Prediction vs reality
- Training data with targets for model validation

---

## Report Workspace

**Purpose:** Compile findings into a structured FYP report

**Chapters:**
1. **Chapter 3:** Methodology - Explains all models and statistical tests
2. **Chapter 4:** Data Collection - Where data came from, cleaning process
3. **Chapter 5:** Results - Model outputs and analysis
4. **Chapter 6:** Portfolio Application - How rebalancing works in practice
5. **Chapter 7:** Discussion - Implications and limitations
6. **Chapter 8:** Conclusions & Recommendations

---

## Key Datasets Produced

| Dataset | Description | Size | Generated In |
|---------|-------------|------|--------------|
| daily_master.csv | Daily KSE-30 + fund data | 1,591 × 22 | Folder 5 (Claude Pipeline) |
| monthly_master.csv | Monthly aggregated | 76 × 30 | Folder 5 (Claude Pipeline) |
| results_rebalancing_forecast.csv | Weight change predictions | ~10K rows | Folder 5+ |
| results_efficiency.csv | Market efficiency test results | ~200 rows | Folder 5+ |

**Note:** These master datasets were created in Folders 4-5. Folder 1b_eda only performed exploratory analysis.

---

## Models Explained

### 1. Fund Flow Prediction
**Purpose:** Predict whether money will flow into or out of KSE-30. Uses mutual fund data (AKD, NBP, NTI) as external variables since these index-tracking funds reflect KSE-30 flows.

**Models Used:**
- **ARIMA (AutoRegressive Integrated Moving Average):** A classical time series model that uses past values to predict future flows. It captures trends and seasonality in the data.
  - How it works: Uses lagged values (past flows) and moving averages to forecast next period's flow
  - Input: Past fund flow percentages, market returns
  - Output: Predicted flow direction (inflow/outflow) and magnitude

- **Random Forest (ML):** An ensemble of decision trees that learns non-linear patterns
  - How it works: Builds multiple trees on different data samples, combines their predictions
  - Input: Market indicators (volume, returns, volatility, macro factors)
  - Output: Binary prediction (inflow/outflow)

---

### 2. GARCH Volatility Modeling
**Purpose:** Model and forecast market volatility (risk) - important for risk management

**Model Used:**
- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity):** Captures volatility clustering (when high volatility follows high volatility)
  - How it works: Models the variance (volatility) as a function of past variances and shocks
  - Equation: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    - ω = constant baseline volatility
    - α = impact of yesterday's shock (news)
    - β = persistence of volatility
  - Input: Past index returns
  - Output: Forecasted volatility for next period

**Why it matters:** Volatility clusters in financial markets - if today is volatile, tomorrow likely is too. GARCH captures this.

---

### 3. Portfolio Optimization
**Purpose:** Determine optimal stock weights to minimize risk for a given return

**Model Used:**
- **Markowitz Mean-Variance Optimization:** The foundational portfolio theory
  - How it works: 
    1. Estimate expected return for each stock
    2. Estimate covariance matrix (how stocks move together)
    3. Find weights that minimize risk for target return
    4. Efficient frontier: best risk-return trade-offs
  - Input: Historical returns, covariance matrix
  - Output: Optimal weights for each stock in portfolio

**Mathematical Formulation:**
```
Minimize: w^T · Σ · w  (portfolio variance)
Subject to: w^T · μ = r_target (expected return)
           Σ w_i = 1 (weights sum to 1)
```

---

### 4. Rebalancing Prediction
**Purpose:** Predict when and how portfolio weights should change to track the index

**Model Used:**
- **Random Forest + Cross-Validation:** Predicts optimal weight changes
  - How it works: 
    1. For each stock, predict weight change based on features
    2. Features: past weights, returns, volume, volatility
    3. Cross-validation ensures model generalizes to unseen data
  - Input: Historical stock weights, returns, volume, technical indicators
  - Output: Predicted weight change for each stock

**Why cross-validation matters:** Prevents overfitting - ensures predictions work on new data, not just training data.

---

### 5. Market Efficiency Tests
**Purpose:** Test if KSE-30 is efficient (prices follow random walk) or predictable

**Tests Used:**

- **Variance Ratio Test:** Checks if returns are random
  - How it works: Compare variance of multi-day returns to single-day returns
  - If ratio ≠ 1, market is predictable (not efficient)
  - Formula: VR(k) = Var(r_t^k) / (k · Var(r_t))
  - If VR(k) > 1: momentum (trends continue)
  - If VR(k) < 1: mean reversion (reversals)

- **Granger Causality Test:** Tests if one variable predicts another
  - How it works: Does past X help predict Y beyond Y's own past?
  - Example: Does oil price Granger-cause KSE-30 returns?
  - If p-value < 0.05, X does Granger-cause Y

- **Augmented Dickey-Fuller (ADF) Test:** Tests stationarity
  - How it works: Tests if series has unit root (non-stationary)
  - If p-value < 0.05, series is stationary (suitable for modeling)
  - Critical for knowing which transformations (log returns) needed

---

### 6. Stationarity Testing (ADF)
**Purpose:** Determine if data is suitable for time series modeling

**Test Used:**
- **Augmented Dickey-Fuller (ADF) Test:**
  - Null hypothesis: Series has a unit root (non-stationary)
  - If p-value < 0.05: Reject null → Series is stationary
  - If p-value > 0.05: Fail to reject → Series is non-stationary

**Why it matters:** ARIMA/GARCH require stationary data. We use log returns (which are typically stationary) instead of price levels (which are typically non-stationary).

---

## Visualizations & Key Insights

This section maps each visualization to the research questions it answers.

### EDA Visualizations (Chapter 5)

| Figure | File | Question Answered |
|--------|------|-------------------|
| **E01 AUM Trend** | `E01_aum_trend.png` | How have fund assets (AKD, NBP, NTI) grown over time? |
| **E02 NAV Return Distribution** | `E02_nav_return_dist.png` | Are fund returns normally distributed? (Answer: No - fat tails) |
| **E03 Fund Flows** | `E03_fund_flows.png` | When did major inflows/outflows occur? |
| **E04 Macro Overview** | `E04_macro_overview.png` | How do oil prices, interest rates, USD/PKR behave? |
| **E05 Monthly Correlation** | `E05_monthly_correlation.png` | Are fund flows correlated with macro variables? (Answer: Weak correlations) |
| **E06 Index Cumulative Return** | `E06_index_cumulative_return.png` | What has been the overall KSE-30 performance? |
| **E07 Top Weights** | `E07_top_weights.png` | Which stocks dominate the index? |

---

### Fund Flow Prediction (Chapter 5)

| Figure | File | Question Answered |
|--------|------|-------------------|
| **FF01 Total Flow Predictions** | `FF01_total_flow_predictions.png` | How accurate are ARIMA/VAR predictions? (Answer: Better than naive, 75% directional accuracy) |
| **FF02 Granger Causality** | `FF02_granger.png` | Do macro variables predict fund flows? (Answer: No single variable dominates) |

---

### GARCH Volatility (Chapter 5)

| Figure | File | Question Answered |
|--------|------|-------------------|
| **G01 Returns & Volatility** | `G01_returns_and_vol.png` | Does volatility cluster? (Answer: Yes - high volatility follows high volatility) |
| **G02 VaR Backtest** | `G02_var_backtest.png` | Is the volatility model calibrated for risk? (Answer: Generally yes) |

---

### Market Efficiency (Chapter 5)

| Figure | File | Question Answered |
|--------|------|-------------------|
| **EF01 ACF** | `EF01_acf.png` | Do returns have serial structure? (Answer: Yes - some autocorrelation) |
| **EF02 Variance Ratio** | `EF02_variance_ratio.png` | Is KSE-30 efficient? (Answer: Mixed evidence - some predictable components) |

---

### Rebalancing Prediction (Chapter 6)

| Figure | File | Question Answered |
|--------|------|-------------------|
| **R01 Retention Probability** | `R01_retention_probability.png` | Which stocks will stay in the index? (AUC: 0.82) |
| **R02 Feature Importances** | `R02_feature_importances.png` | What predicts rebalancing? (Answer: Current weight, size, stability) |
| **R03 Weight Scatter** | `R03_weight_scatter.png` | How well do we predict weights? (R²: 0.97) |
| **R04 Weight Changes** | `R04_weight_changes.png` | Which stocks will gain/lose weight? |

---

### Summary Dashboard

| Figure | File | Question Answered |
|--------|------|-------------------|
| **Summary Dashboard** | `SUMMARY_dashboard.png` | What is the overall project performance summary? |

---

## Presentation Summary

### The Problem We Solved
We analyze KSE-30 (Pakistan's top 30 index). The key questions:
1. **Predict 3-fund composite KSE-30 tracker flow** - Using mutual fund data from AKD, NBP, and NTI
2. **Manage volatility** - How risky is the market?
3. **Rebalance portfolios** - When should stocks be bought/sold to track the index?

### Our Solution
Built an automated pipeline with 7 core models:

```
Data → Preprocessing → Fund Flow Prediction
                      → GARCH Volatility
                      → Portfolio Optimization
                      → Rebalancing Prediction
                      → Market Efficiency Tests
                      → Backtesting
```

### Results
- Since AKD, NBP, and NTI track KSE-30, the project predicts an AUM-weighted 3-fund composite-flow series based on market indicators
- GARCH models capture volatility clustering
- Portfolio optimization provides optimal weights
- Rebalancing predictions with cross-validation
- Market efficiency tests show predictive potential
- Backtested on March 2026 data

---

## Data Sources Used

### 1. Stock Market Data (KSE-30)

| File | Description | Columns |
|------|-------------|---------|
| `kse30_daily_data.csv` | Daily KSE-30 constituent data | Date, ISIN, Symbol, Company, Price, Index Weight %, FF Shares, FF MCAP, Ord Shares, Ord MCAP, Volume |
| `kse30_stocks_clean.csv` | Cleaned stock-level data | Date, Symbol, Company, Price, Weight %, FF Shares, FF MCAP, Volume, Log Return, Rolling Vol, MA20, MA50 |

**Source:** Pakistan Stock Exchange (PSX)
**Date Range:** 2020-01-01 to 2026-04-30
**Coverage:** 30 constituent stocks

---

### 2. Mutual Fund Data (NAV & AUM)

| File | Sheets | Columns |
|------|--------|---------|
| `funds_data.xlsx` | AKD, NBP, NTI | DATE, NAV, AUM |

**Funds Analyzed:**
- **AKD** - AKD Index Tracking Fund
- **NBP** - NBP Balanced Index Fund
- **NTI** - National Index Fund

**What We Used:**
- **NAV** (Net Asset Value) - Daily fund prices
- **AUM** (Assets Under Management) - Total assets

**How Fund Flows Were Derived:**
Since these funds track KSE-30, we used NAV and AUM data to calculate fund flows:
- Fund Flow = Monthly change in AUM
- Flow % = (Flow / Previous AUM) × 100
- NAV Returns = Daily/monthly fund returns

These flows are combined into an AUM-weighted 3-fund composite-flow series for forecasting and analysis. They should not be described as an official observed KSE-30 net-flow series.

---

### 3. Macro Economic Data

| Source | File | Columns |
|--------|------|---------|
| Oil Prices | macro_data.xlsx (OIL) | DATE, PRICE (Brent) |
| Interest Rates | macro_data.xlsx (IR) | DATE, RATE (SBP policy rate) |
| USD/PKR Exchange | macro_data.xlsx (USD) | DATE, USD |
| CPI | cpi.csv | Period, YoY CPI (%) |
| Gold | gold.csv | Date, Price, Open, High, Low, Volume, Change % |

**Date Range:** 2018-07 to 2026-02

---

### 4. Master Datasets Created

| Dataset | Description | Columns |
|---------|-------------|---------|
| `daily_master.csv` | Daily merged market + fund + macro | 1,301 rows × 22 columns: date, idx_total_volume, idx_n_stocks, idx_ff_mcap_total, idx_log_return, idx_rolling_vol_30d, oil_price, oil_log_return, usdpkr, usdpkr_log_return, interest_rate, cpi_yoy, nav_akd, nav_return_akd, nav_vol_akd, nav_nbp, nav_return_nbp, nav_vol_nbp, nav_nti, nav_return_nti, nav_vol_nti |
| `monthly_master.csv` | Monthly aggregated | 60 rows × 30 columns: date, oil_price_end, oil_return_monthly, usdpkr_end, usdpkr_return_monthly, interest_rate_end, cpi_yoy_end, idx_return_monthly, idx_vol_monthly, nav_akd_end, nav_return_akd_monthly, aum_akd, flow_akd, flow_pct_akd, flow_spike_akd, nav_nbp_end, nav_return_nbp_monthly, aum_nbp, flow_nbp, flow_pct_nbp, flow_spike_nbp, nav_nti_end, nav_return_nti_monthly, aum_nti, flow_nti, flow_pct_nti, flow_spike_nti, total_fund_flow |

---

### 5. Result Datasets

| File | Description |
|------|-------------|
| `results_fund_flow.csv` | Fund flow prediction model performance |
| `results_garch.csv` | GARCH model parameters per fund |
| `results_efficiency.csv` | Market efficiency test results |
| `results_rebalancing.csv` | Actual vs predicted rebalancing weights |
| `results_rebalancing_forecast.csv` | Predicted weight changes with features |

---

### Data Summary

| Category | Sources | Key Variables |
|----------|---------|---------------|
| **Market** | PSX (KSE-30) | Prices, volumes, weights, returns |
| **Funds** | AKD, NBP, NTI | NAV, AUM, flows |
| **Commodities** | Brent Oil | Prices, returns |
| **Macro** | SBP, CPI | Interest rate, inflation |
| **Currency** | USD/PKR | Exchange rate |

**Total Period:** 2020-2026 (daily/monthly data)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Time Series Models | statsmodels (ARIMA, GARCH) |
| Machine Learning | scikit-learn (Random Forest) |
| Statistics | statsmodels (ADF, Granger, Variance Ratio) |
| Reporting | Markdown, PNG |

---

## Future Improvements

### Data Enhancements
- **More funds:** Include more mutual funds beyond AKD, NBP, NTI
- **Longer history:** Extend dataset back further for more training data
- **Alternative data:** Add sentiment analysis from news, social media

### Model Improvements
- **Deep learning:** Use LSTM or Transformer models for better time series forecasting
- **Ensemble methods:** Combine multiple models (ARIMA + GARCH + ML) for robust predictions
- **Real-time updates:** Set up automated data pipelines for live predictions

### Feature Engineering
- **Technical indicators:** Add RSI, MACD, Bollinger Bands
- **Macro factors:** Include GDP, industrial production, trade balance
- **Market microstructure:** Add order book data, bid-ask spreads

### Application Extensions
- **Risk management:** Integrate VaR, CVaR for better risk assessment
- **Backtesting framework:** Build a full backtesting system with transaction costs
- **Web dashboard:** Create interactive dashboard for fund managers
- **API service:** Expose predictions via REST API

---

## Q&A - Anticipated Questions

### General Questions

**Q: What is the goal of this project?**
A: This project builds an automated system to analyze and predict KSE-30 index behavior. Since AKD, NBP, and NTI funds track KSE-30, we use their flows as external variables - flows in these index-tracking funds reflect flows in KSE-30 itself.

**Q: Why KSE-30? Why Pakistani market?**
A: KSE-30 is the top 30 stocks on Pakistan Stock Exchange (PSX). It's an important index for institutional investors. This project fills a gap in quantitative finance research on emerging markets like Pakistan.

**Q: What data did you use?**
A: 
- Daily stock prices for KSE-30 constituents (2020-2026)
- Mutual fund NAVs (AKD, NBP, NTI)
- Macro data: CPI, interest rates, oil prices, USD/PKR exchange rate

---

### Model-Specific Questions

**Q: Why use both ARIMA and Random Forest for fund flow prediction?**
A: ARIMA captures linear temporal patterns and trends in historical flows. Random Forest captures non-linear relationships between market indicators and flows. Using both gives more robust predictions.

**Q: What is volatility clustering? Why does GARCH capture it?**
A: Volatility clustering means high volatility periods tend to follow high volatility periods (and vice versa). GARCH explicitly models this by making today's variance depend on yesterday's shock and variance - perfect for financial markets.

**Q: What is the efficient frontier in portfolio optimization?**
A: The efficient frontier shows the best possible risk-return combinations. Any portfolio on this frontier gives maximum return for its risk level. Investors choose their optimal point based on risk tolerance.

**Q: Why test market efficiency?**
A: If the market is perfectly efficient (random walk), predictions are impossible. Our tests show the market has predictable components, justifying our modeling approach.

**Q: What does Granger Causality tell us?**
A: It doesn't prove true causation, but shows predictive causality: "Does knowing X's past help predict Y's future?" For example, if oil prices Granger-cause KSE-30 returns, we can use oil as a predictor.

**Q: Why use log returns instead of prices?**
A: Prices are non-stationary (they trend). Log returns are typically stationary (mean-reverting), making them suitable for ARIMA/GARCH models. Returns are also additive and more statistically tractable.

---

### Technical Questions

**Q: What is cross-validation and why use it?**
A: Cross-validation trains on a subset of data and tests on the held-out portion. It prevents overfitting - making sure the model works on new data, not just memorized training data.

**Q: How do you handle missing data?**
A: We use forward-fill for macro data (carry last known value forward). Stationarity tests help identify which transformations are needed.

**Q: How accurate are the predictions?**
A: We backtested on March 2026 data (held out from training). The models show predictive potential, though accuracy varies by model and market conditions. More data would improve results.

**Q: What libraries did you use?**
A: Python with Pandas/NumPy (data), statsmodels (ARIMA, GARCH, ADF, Granger), scikit-learn (Random Forest), Matplotlib/Seaborn (visualization).

---

### Limitations & Improvements

**Q: What are the limitations of this project?**
A:
- Limited data history (2020-2026)
- Only 3 funds analyzed
- Backtesting on single month
- No transaction costs included
- Market conditions change over time

**Q: How could this be improved?**
A:
- Add more funds and longer history
- Use deep learning (LSTM) for better predictions
- Include technical indicators (RSI, MACD)
- Build real-time dashboard
- Full backtesting with transaction costs

**Q: Is this ready for real-world use?**
A: It's a proof-of-concept. For production use, we'd need:
- More rigorous backtesting
- Real-time data pipelines
- Risk management (VaR, CVaR)
- Integration with fund management systems

---

### Business/Practical Questions

**Q: How can fund managers use this?**
A: 
- Anticipate inflows/outflows to manage cash
- Use volatility forecasts for risk management
- Rebalancing predictions to minimize tracking error
- Efficiency tests to validate strategies

**Q: What makes this project unique?**
A: Quantitative finance research on Pakistan market is scarce. This provides baseline models for KSE-30 analysis that can be built upon.

**Q: What's the main takeaway?**
A: The KSE-30 market shows predictable patterns. By combining traditional finance (Markowitz) with modern ML (Random Forest), we can build useful prediction systems for emerging markets.

---

## Key Points We Missed

Based on industry best practices for KSE-30 fund flow prediction, here are the key factors and data sources we could have included:

### 1. Fund Flow Types We Didn't Distinguish
- **Foreign Portfolio Investment (FPI/FIPI)** - Most impactful on KSE-30 movement
- **Local Institutional (LIPI)** - Mutual funds, insurance, pension funds, banks
- **Retail flows** - High in Pakistan but volatile

We used mutual fund data broadly but didn't separate by investor category.

### 2. Data Sources We Missed
| Source | What It Provides |
|--------|------------------|
| **NCCPL** | Daily/weekly FIPI & LIPI data (best for fund flows) |
| **MUFAP** | Mutual fund data by category |
| **Broker reports** (Topline, AKD, Arif Habib) | Weekly flow summaries |
| **SBP** | Quarterly foreign portfolio investment aggregates |

### 3. Key Predictive Factors We Didn't Include
| Factor | Impact | Why It Matters |
|--------|--------|----------------|
| **Political Stability** | Very High | Pakistan highly sensitive to political changes |
| **Exchange Rate (PKR/USD)** | Strong | Depreciation triggers outflows |
| **Interest Rates / T-bills** | High | Higher rates attract money from stocks |
| **IMF Program / External Aid** | Very High | Positive for inflows |
| **Inflation & Remittances** | Medium-High | Remittances support market liquidity |
| **Global Risk Sentiment (VIX)** | High | Risk-off = FPI outflows |
| **Market Valuation (P/E)** | Medium | Low valuations attract flows |
| **Past Returns (Momentum)** | High | Performance chasing behavior |

### 4. Prediction Methods We Could Have Added
- **VAR (Vector Auto Regression)** - Model interdependencies between flows, returns, and macro variables
- **GARCH-MIDAS** - Combine macro data with GARCH for mixed-frequency modeling
- **XGBoost** - Better than Random Forest for structured tabular data
- **Sentiment Analysis** - Google Trends for "PSX", news sentiment scoring

### 5. Practical Improvements
- Create a composite "Flow Score" combining weighted macro + sentiment factors
- Monitor weekly FIPI trends from broker reports
- Focus on heavy-weight stocks in KSE-30 (banks, oil & gas, fertilizers) for flow impact

### 6. Limitations to Acknowledge
- Fund flows are partly **unpredictable** in short term due to sudden news (politics, geopolitics)
- Foreign flows in Pakistan are often "hot money" - very sensitive to global conditions
- Local mutual fund flows tend to be more stable but still influenced by past performance

---

## How to Improve (Updated)

Building on what we missed:

1. **Source better flow data** - NCCPL for FIPI/LIPI, MUFAP for mutual funds
2. **Add macro predictors** - Exchange rate, interest rates, political risk index, remittances
3. **Use better models** - VAR, GARCH-MIDAS, XGBoost
4. **Add sentiment** - News sentiment, Google Trends for "Pakistan stocks"
5. **Separate by investor type** - Foreign vs local institutional vs retail

---

*Project completed: June 2026*
