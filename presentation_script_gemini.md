Here is a presentation script tailored for your defense or final presentation. It is designed to sound natural, mathematically sound, and transitions smoothly across slides 13 to 18.

---

### Slide 13: Fund Flow Forecasting Results

**"Moving into our empirical results, let’s first look at Fund Flow Forecasting.**

To predict net movements into our target KSE-30 index-tracking funds, we evaluated two dynamic models—ARIMAX and a Vector Autoregression model—against a naive Random Walk benchmark.

As you can see from the table, our **ARIMAX(1,0,1) model significantly outperformed the baseline**, reducing the Root Mean Squared Error by 29.5%, bringing it down from over 83 to 58.56.

But what is highly critical for executing actual trading or asset management strategies is **Directional Accuracy**. Both our ARIMAX and VAR models achieved a **75% directional accuracy rate**, compared to just 37.5% for the naive benchmark.

We also ran Granger causality tests on our macroeconomic variables, and no single variable passed at the 5% level. This empirically confirms that macro factors do not influence flows in isolation; rather, a multivariate approach is absolutely necessary.

Our core takeaway here is that while these models face challenges as strict point estimators due to emerging market noise, they are **highly powerful, robust tools for directional forecasting**."

---

### Slide 14: Fund Flow Forecasting (Visualizations)

**"To give you a better visual sense of these forecasting dynamics, let’s look at the plots.**

The top chart illustrates the combined monthly net flows for the AKD, NBP, and NTI index-tracking funds from January 2021 onwards. You'll immediately notice a massive structural spike in December 2025 and early 2026, where aggregate net inflows peaked near 234 million PKR.

The bottom chart displays our out-of-sample testing window starting from early 2024, plotting the actual test data in black against our ARIMAX and VAR fits.

While the VAR model—represented by the green line—yielded a higher $R^2$ of 0.387 on paper, it tends to over-smooth the extreme volatility blocks. Meanwhile, the ARIMAX model—in red—captures the directional shifts more dynamically. This visualization emphasizes why we treat these outputs primarily as directional regime filters rather than exact point estimators."

---

### Slide 15: Volatility Modeling Results

**"Next, we look at the risk and volatility dynamics of the KSE-30 using GARCH and EGARCH frameworks.**

When modeling daily returns, our traditional $GARCH(1,1)$ yielded an $(\alpha + \beta)$ persistence parameter of **0.9667**. This exceptionally high persistence indicates that volatility shocks in the Pakistani market decay very slowly; when a shock hits the PSX, the uncertainty lingers for an extended period.

However, our preferred model is the **EGARCH(1,1)**, because it achieved a lower Akaike Information Criterion of 4209.68 and revealed a statistically significant, negative gamma term. This **confirms a strong leverage effect in the KSE-30**. In plain terms: bad news or market drops trigger a disproportionately larger spike in volatility than an equivalent amount of good news.

To validate the real-world utility of this model, we conducted a **Value-at-Risk (VaR) Backtest** at a 5% risk threshold across 1,300 historical observations. Theoretically, we expect about 5% of trading days to breach this downside threshold. Our model recorded exactly 58 exceedances—which translates to **4.46%**—validating that our EGARCH-based VaR is incredibly tight and reliable for institutional risk calibration."

---

### Slide 16: Volatility Modeling (Visualizations)

**"This slide visually maps that volatility journey and the accuracy of our risk model.**

The top panel plots the daily log returns of the KSE-30 against the conditional volatility derived from our EGARCH model. You can clearly see the classic financial phenomenon of **volatility clustering**—especially around the sharp market downturn on March 2nd, 2026, where daily returns plunged by over 10%.

The bottom panel highlights our 5% Value-at-Risk backtest. The solid black line represents our dynamic downside risk boundary. Notice how the VaR line intelligently opens up and widens during high-volatility regimes to protect the portfolio, and tightens during calmer periods.

Out of 1,300 daily observations, the actual returns breached this boundary only 58 times. Sitting at a 4.46% error rate against a 5% nominal target, this proves our econometric framework can serve as an excellent downside guardrail for local asset managers."

---

### Slide 17: Market Efficiency Results

**"This brings us to a foundational question of our research: *Is the KSE-30 market efficient?***

The short answer is: **the results are mixed.** Our empirical testing indicates that the KSE-30 is neither fully efficient nor fully inefficient, but rather occupies a complex middle ground.

Let's break down the metrics:

* Our **Runs Test** yielded a Z-statistic of $-1.91$ with a p-value of 0.0561, making it borderline efficient.
* Similarly, the **Variance Ratio Test** at a short-horizon $VR(2)$ resulted in a statistic of 1.0069 with a p-value of 0.8752, meaning we cannot reject the random walk hypothesis at short intervals.

However, when we look at the **Ljung-Box Q-test**, the p-value drops to 0.0000, revealing strong serial autocorrelation. Furthermore, the **Hurst Exponent** came out to **0.6559**. Because $H > 0.5$, this is clear empirical proof of **long-memory behavior and persistence** in the time series.

While prices incorporate basic information relatively quickly at a glance, the strong serial dependence and long-memory structure prove that **exploitable statistical patterns do exist** in the PSX, allowing for active, predictive strategies."

---

### Slide 18: Portfolio Rebalancing: Framework

**"Leveraging these exploitable market anomalies, we designed an institutional Portfolio Rebalancing Framework to predict index recomposition.**

As we know, the KSE-30 index undergoes semi-annual rebalancing. For an index-tracking fund or an active quant strategy, anticipating these shifts before they happen is a massive alpha generator.

We compiled a dataset of **422 stock-window observations across 16 semi-annual rebalancing periods**, extracting features like historical weights, 30, 60, and 90-day momentum, daily volatility, average trading volume, and weight drift.

Using this panel data, we set up **two distinct prediction tasks**:

1. **Exclusion/Inclusion Prediction:** A binary classification task to determine whether a constituent stock will survive or be kicked out of the index in the next review window.
2. **Weight Prediction:** A regression task to forecast the exact percentage weight a stock will hold post-recomposition.

This pipeline essentially acts as a forward-looking decision engine, and on the next slide, we will see how our machine learning models performed on these tasks."

---

### Terms Explained

#### RMSE

**RMSE** means Root Mean Squared Error.

It measures how far predictions are from actual values, with larger errors receiving more weight.

Lower RMSE means better forecast performance.

#### Directional Accuracy

Directional Accuracy measures how often the model correctly predicts the **direction** of movement rather than the exact value.

In this project, it means correctly predicting whether fund flow moves toward **inflow** or **outflow**.

#### Random Walk Benchmark

The Random Walk benchmark is the simplest baseline model.

It assumes the next period behaves much like the current one.

If ARIMAX or VAR outperform it, that means they are extracting useful predictive structure.

#### ARIMAX

**ARIMAX** stands for Autoregressive Integrated Moving Average with Exogenous variables.

It uses:

- past values of the target variable
- past forecast errors
- outside explanatory variables such as macroeconomic indicators

#### ARIMAX(1,0,1)

The notation `ARIMAX(1,0,1)` tells us the model structure:

- the first `1` means one autoregressive lag
- the `0` means no differencing
- the last `1` means one moving-average lag

So the model uses:

- one previous value of the target series
- one previous error term
- exogenous variables such as macroeconomic indicators

In simple terms, it uses recent fund-flow history plus outside macro information to forecast the next flow movement.

#### VAR

**VAR** stands for Vector Autoregression.

It models several variables together, allowing each one to depend on its own lags and the lags of the others.

That is useful when flows and macro variables influence each other jointly.

#### Granger Causality

Granger causality checks whether past values of one variable help predict another variable.

It does not prove true economic causation. It only tests predictive usefulness through lags.

#### R-squared

`R²` measures how much variation in the target is explained by the model.

Higher values usually mean a better fit, but in your flow section the more important metric is often directional accuracy rather than exact magnitude fit.

#### GARCH

**GARCH** stands for Generalized Autoregressive Conditional Heteroskedasticity.

It is used to model time-varying volatility in financial returns.

It is useful because market risk is not constant over time.

#### GARCH(1,1)

The notation `GARCH(1,1)` means:

- the first `1` is one lag of squared shocks
- the second `1` is one lag of past variance

So current volatility depends on:

- yesterday's shock
- yesterday's volatility

This is what allows the model to capture volatility clustering.

#### EGARCH

**EGARCH** stands for Exponential GARCH.

It extends GARCH by allowing **asymmetric** effects, meaning negative and positive shocks can affect future volatility differently.

#### EGARCH(1,1)

The notation `EGARCH(1,1)` means:

- one lag of the shock effect
- one lag of past log-variance

It keeps a similar lag structure to GARCH, but it models the log of variance and allows asymmetric responses.

This means it can capture the fact that bad news may increase future volatility more than equally sized good news.

#### Alpha and Beta

In GARCH:

- **alpha** measures how strongly volatility reacts to recent shocks
- **beta** measures how strongly past volatility carries forward

When **alpha + beta** is close to 1, volatility is highly persistent.

#### Persistence

Persistence means volatility shocks decay slowly rather than disappearing quickly.

In your result, `alpha + beta = 0.9667`, which means once volatility rises, it tends to remain elevated for some time.

#### AIC

**AIC** means Akaike Information Criterion.

It is used to compare competing models.

Lower AIC suggests a better balance between fit and simplicity.

#### Gamma

In EGARCH, **gamma** is the asymmetry parameter.

A significant negative gamma means bad news increases future volatility more than equally sized good news.

#### Leverage Effect

The leverage effect means negative shocks create a stronger volatility response than positive shocks of the same size.

In plain words, bad news disturbs the market more than good news stabilizes it.

#### Value at Risk (VaR)

**VaR** means Value at Risk.

It gives a downside loss threshold.

For a 5% VaR model, we expect about 5% of returns to be worse than the VaR cutoff.

#### VaR Exceedance

An exceedance occurs when the actual loss is worse than the VaR threshold.

Your result of 58 exceedances out of 1,300 observations is close to the expected 5%, which supports the model’s calibration.

#### Volatility Clustering

Volatility clustering means high-volatility periods are often followed by high-volatility periods, while calm periods are followed by calm periods.

This is a very common pattern in financial markets.

#### Weak-Form Market Efficiency

Weak-form efficiency means past prices and returns are already reflected in current prices.

If a market is fully weak-form efficient, past return patterns should not help predict future returns.

#### Random Walk Hypothesis

The random walk hypothesis says price changes are not systematically predictable from past price movements.

If this holds strictly, active strategies based only on historical price behavior should not consistently outperform.

#### Runs Test

The Runs Test checks whether positive and negative returns occur in a random sequence.

If they do, that supports random-walk behavior.

#### Variance Ratio Test

The Variance Ratio test checks whether return variance scales over time in a way that matches a random walk.

Values close to 1 generally support efficiency.

#### Ljung-Box Q-Test

The Ljung-Box Q-test checks for serial correlation across several lags.

A very small p-value means the series is not fully independent over time.

#### Hurst Exponent

The Hurst exponent, written as `H`, measures long-memory behavior.

- `H = 0.5` suggests random walk
- `H > 0.5` suggests persistence
- `H < 0.5` suggests mean reversion

Your result of `0.6559` suggests persistent long-memory structure.

#### Portfolio Rebalancing

Portfolio rebalancing means adjusting holdings when index membership or stock weights change.

In your project, the goal is to anticipate these changes before the official KSE-30 review.

#### Inclusion/Exclusion Prediction

This is a classification task that predicts whether a stock will remain in the index or be removed in the next rebalancing window.

#### Weight Prediction

This is a regression task that predicts the stock’s future index weight after recomposition.

#### Momentum

Momentum means recent price strength or weakness over a chosen horizon, such as 30, 60, or 90 days.

#### Weight Drift

Weight drift means how much a stock’s current index weight has changed since the previous recomposition.

#### Market-Cap Proxy

A market-cap proxy is an estimated measure of company size, usually based on shares outstanding multiplied by price.
