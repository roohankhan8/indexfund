# Presentation Script — Slides (Pages) 13 to 18

**Pages 13–18 (from the provided PDF):**
- Page 13: Exploratory Data Analysis (key EDA statistics)
- Page 13/14: Fund Flow Forecasting Results (Naive vs ARIMAX vs VAR)
- Page 15: Volatility Modeling Results (GARCH / EGARCH + persistence + VaR)
- Page 16: (Volatility / VaR validation continuation)
- Page 17: Market Efficiency Results (Runs / Variance Ratio / Ljung–Box / Hurst)
- Page 18: Market Efficiency summary (interpretation)

**Time:** ~2.5–4 minutes total

---

## Slide/Page 13 — Exploratory Data Analysis (EDA)
“On slide 13, we summarize what the data looks like and what that implies for modeling.

First, the relationship between *contemporaneous* aggregate flow and KSE-30 returns is extremely small—around **0.0095**. This tells us that we cannot rely on a simple same-month correlation. Instead, the signal—if it exists—must work through **lagged dynamics**, which justifies our ARIMAX and VAR structures.

Second, the monthly fund flow series is **episodic**. The mean is about **13.61 mn PKR**, but the distribution has large outliers: the peak inflow is roughly **234.86 mn PKR** and the peak outflow is around **-44.01 mn PKR**. So the market experiences inflow/outflow episodes rather than smooth behavior.

Third, **AUM is highly concentrated**, meaning AKD dominates the aggregate sector AUM. That affects interpretation: the aggregate sector-flow proxy is strongly influenced by the largest fund.

Finally, the return series is heavy-tailed: Jarque–Bera p-value is effectively zero, so returns are non-normal and we see **fat tails**. This is exactly why we move to volatility models like **GARCH/EGARCH**, because constant-variance assumptions would be unrealistic.”

---

## Slide/Page 13–14 — Fund Flow Forecasting Results (ARIMAX / VAR)
“Now we show whether those EDA-driven expectations translate into forecastable patterns.

We compare a **naive random-walk benchmark** against two dynamic models:

- Naive: directional accuracy **37.5%**
- **ARIMAX(1,0,1)**: directional accuracy **75.0%**, and it reduces forecasting errors substantially
- **VAR(1)**: also **75.0%** directional accuracy

Two important interpretation points:

1) The **R² can remain negative**—that’s because predicting the exact PKR magnitude of flows is difficult with noisy, shock-driven data.

2) The project is strongest in the **directional metric**: getting the sign right (inflow vs outflow) is materially better than naive.

So our conclusion from this slide is: fund-flow magnitude is hard, but flow *direction* becomes useful for decision-making.”

---

## Slide/Page 15 — Volatility Modeling Results (GARCH / EGARCH + Persistence)
“Next, we move from ‘direction’ to ‘risk’.

We fit **GARCH(1,1)** and **EGARCH(1,1)** on the reconstructed KSE-30 daily returns.

Key result: volatility persistence is high. In GARCH terms, **alpha + beta ≈ 0.9667**, which means shocks decay slowly. If the market becomes volatile, it tends to stay volatile for a while.

Then EGARCH adds the crucial financial realism: we see a significant **negative gamma**, which confirms a **leverage effect**. In other words, **bad news increases volatility more strongly** than good news of the same magnitude.

So at a portfolio level, this means risk is not stable; it responds asymmetrically to negative shocks.”

---

## Slide/Page 15–16 — VaR Backtesting (risk validation)
“After estimating volatility, we validate it using a VaR backtest.

We run a **5% VaR backtest** and observe **58 exceedances out of 1,300 observations**, which is about **4.46%**.

That’s close to the nominal 5%, so we can say the volatility model produces a reasonably calibrated downside risk envelope.

This supports our later strategy design: we can modulate position sizing based on conditional volatility rather than using a constant risk assumption.”

---

## Slide/Page 17 — Market Efficiency Results (Runs / VR / Ljung–Box / Hurst)
“Now we test market efficiency—because if everything were purely random, forecasting and tilting would have little justification.

The efficiency evidence is **mixed**, which is a common pattern in emerging markets.

- Runs test: **borderline efficient** (p ≈ **0.0561**), meaning sign-randomness is close to the efficient benchmark
- Variance Ratio (VR): at VR(2) and VR(4), the p-values suggest efficiency is not strongly rejected
- Ljung–Box Q: p ≈ **0.0000**, indicating significant serial correlation—so it contradicts full random-walk behavior
- Hurst exponent: **H = 0.6559**, meaning persistence and long-memory behavior

So the combined interpretation is: short-horizon behavior can resemble random walk, but longer-horizon/structure-based tests reveal persistence.”

---

## Slide/Page 18 — Efficiency Summary (one clean takeaway)
“This slide summarizes the takeaway:

KSE-30 is not purely efficient and not purely inefficient. Instead, it behaves like an emerging market with *pockets of predictability*.

That mixed efficiency result is exactly what motivates our overall project: we use forecasting models for flows and a rebalancing framework for index constituents, because there is evidence of exploitable structure—even if it’s not cleanly detectable by a single test.”

---

## One-line closing for this section
“From pages 13–18, we conclude: flow direction is forecastable, volatility is persistent and asymmetric, and market efficiency is mixed—so rebalancing using probabilistic signals is justified.”

