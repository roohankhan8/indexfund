# Presentation Script — Slides 13 to 18

This script is for the section you present tomorrow. It follows the actual slide numbering in `Analyzing Fundflow patterns - Presentation.pdf.pdf`, covering slides **13, 14, 15, 16, 17, and 18**.

Estimated speaking time: about 4 to 5 minutes.

## Slide 13 — Fund Flow Forecasting Results

"This slide shows how well we can forecast aggregate KSE-30 sector fund flows.

We compare three approaches: a naive random-walk benchmark, ARIMAX(1,0,1), and VAR(1).

The naive model performs the weakest, with an RMSE of 83.10, MAE of 51.54, and directional accuracy of only 37.5%.

ARIMAX performs best overall. It reduces RMSE to 58.56, lowers MAE to 37.35, and improves directional accuracy to 75%.

VAR also performs much better than the naive benchmark, with 75% directional accuracy, although its forecast errors are slightly higher than ARIMAX.

So the main takeaway is that dynamic multivariate models clearly outperform the simple benchmark, especially when the objective is to identify whether flows are likely to move into an inflow or outflow regime."

## Slide 14 — Fund Flow Forecasting Results Figure

"This slide visually supports the previous result.

The key point here is that exact monthly PKR amounts are difficult to predict because fund flows are irregular and shock-driven. They react to market stress, macro news, investor sentiment, and timing effects in subscriptions and redemptions.

So even if the predicted line does not perfectly match the exact magnitude every month, the important result is that ARIMAX and VAR capture the direction of movement much better than the naive model.

This is also consistent with our earlier correlation and Granger causality findings. No single macroeconomic variable independently explains flows at the 5% level, so we need models that capture combined and lagged interactions rather than isolated one-variable effects.

That is why our conclusion is that these models are stronger as directional tools than as exact point estimators."

## Slide 15 — Volatility Modeling Results

"After forecasting flows, we move to risk modeling.

Here we estimate GARCH(1,1) and EGARCH(1,1) on reconstructed KSE-30 daily returns.

The GARCH model shows high volatility persistence, because alpha plus beta equals 0.9667. This means shocks decay slowly, so when volatility rises, it tends to remain elevated for some time.

However, EGARCH is the preferred model because it has the lower AIC, 4209.68 compared to 4243.43 for GARCH.

More importantly, EGARCH captures the leverage effect through a significant negative gamma term. This tells us that bad news creates a larger increase in future volatility than equally sized good news.

So the broader conclusion is that KSE-30 risk is not constant. It is persistent, time-varying, and more sensitive to negative shocks."

## Slide 16 — VaR Backtesting

"This slide validates the volatility model using Value-at-Risk backtesting.

For a 5% VaR model, we expect roughly 5% of observations to breach the VaR threshold.

In our case, there are 58 exceedances out of 1,300 observations, which is 4.46%.

That is very close to the expected 5%, so the model appears reasonably well calibrated.

This matters because it means the volatility model is not only statistically fitted, but also practically useful as a downside-risk envelope.

So we can use this EGARCH-based volatility estimate as a risk overlay in the broader investment and rebalancing framework."

## Slide 17 — Market Efficiency Results

"Next, we test weak-form market efficiency in the KSE-30 using multiple diagnostics.

The runs test gives a p-value of 0.0561, so it is borderline efficient.

The variance ratio result at short horizons is also broadly consistent with random-walk behavior.

But the story changes when we look at the Ljung-Box test and the Hurst exponent.

The Ljung-Box p-value is effectively zero, which indicates significant serial dependence, and the Hurst exponent is 0.6559, which points to persistence and long memory.

So the market is not fully efficient and not fully inefficient either.

Our interpretation is mixed efficiency: short-horizon behavior sometimes looks close to a random walk, but longer-horizon dependence and memory still exist."

## Slide 18 — Portfolio Rebalancing Framework

"This final slide in my section shows how the earlier findings are translated into a practical framework.

We define two prediction tasks.

The first is inclusion prediction: whether a stock will remain in the KSE-30 after the next recomposition.

The second is weight prediction: what the constituent weight will be.

The dataset contains 422 stock-window observations across 16 semi-annual rebalancing periods, and the features include current weight, 30-, 60-, and 90-day momentum, volatility, average volume, market-cap proxy, price-to-moving-average ratios, weight drift, and weight range.

The reason this framework is important is that it connects the earlier empirical findings to an actionable decision-support tool.

Fund-flow direction gives the regime signal, volatility modeling gives the risk overlay, and the rebalancing module helps identify which constituents are likely to stay, leave, gain weight, or lose weight.

So this is the bridge from pure analysis into portfolio positioning ahead of official index reviews."

## Closing Bridge to the Next Presenter

"So to summarize my section: fund flows are directionally predictable, KSE-30 volatility is persistent and asymmetric, market efficiency is mixed, and these findings justify a predictive rebalancing framework. From here, we move into the detailed rebalancing results and strategy implications."
