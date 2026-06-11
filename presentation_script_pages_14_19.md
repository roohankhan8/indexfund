# Presentation Script — Slides/Pages 14 to 19 (Shifted +1 from the earlier draft)

**Block covers (as per the provided PDF):**
- Slide/Page 14–15: Fund Flow Forecasting Results (Naive vs ARIMAX vs VAR)
- Slide/Page 15–16: Volatility Modeling Results (GARCH/EGARCH persistence & leverage)
- Slide/Page 17: VaR Backtesting
- Slide/Page 17–18: Market Efficiency Results (Runs, Variance Ratio, Ljung–Box, Hurst)
- Slide/Page 19: Efficiency takeaway / narrative summary into the rebalancing section

**Estimated speaking time:** ~3–4 minutes.

---

## Slide/Page 14 — Fund Flow Forecasting Results (Naive vs ARIMAX vs VAR)
“Now we test whether the flow signal is usable.

We compare three models against a naive random-walk benchmark:
- **Naive (RW):** directional accuracy **37.5%**
- **ARIMAX(1,0,1):** directional accuracy improves to **75.0%**
- **VAR(1):** also achieves **75.0%** directional accuracy

Key interpretation:
- Flow magnitude is noisy and shock-driven, so **R² can be negative**.
- But direction—whether flows are expected to be inflow vs outflow—improves substantially.

This is why, in the decision-making process, we treat flows primarily as a **direction/regime tool**, not as an exact PKR point estimator.”

---

## Slide/Page 15 — Fund Flow Forecasting Results (what the plot/table implies)
“On the corresponding results figure, you should emphasize the narrative:

Even when the predicted line doesn’t perfectly match the exact magnitude month-to-month, the dynamic models capture the **inflow/outflow episodes better** than the naive baseline.

### Why fund-flow graphs spike (quick intuition)
Those sharp spikes usually happen when **NAV moves** and investors/fund managers react through **subscriptions and redemptions**—often during **macro/policy surprises** (rates, inflation, FX), or during **market stress/crises** when investors rebalance quickly. Since our “flow” is derived from AUM changes adjusted by NAV, any sudden NAV move or reporting/timing effect can also make the computed flow look abrupt.

We also justify this from the earlier EDA:
- contemporaneous correlation is tiny,
- so the predictive structure must come from **lagged relationships**.

That’s exactly what ARIMAX and VAR are designed to exploit.”

---

## Slide/Page 15 — Volatility Modeling Results (GARCH vs EGARCH)
“Next we move from predicting flows to modeling **risk**.

We fit **GARCH(1,1)** and **EGARCH(1,1)** on reconstructed KSE-30 daily returns.

Two core results:
1) **Persistence:** volatility shocks decay slowly. In GARCH terms, alpha + beta is about **0.9667**.
   - Practically: when the market becomes volatile, it tends to remain volatile.

2) **Asymmetry / leverage effect:** EGARCH shows a significant **negative gamma**, meaning **bad news** increases future volatility more than good news of the same magnitude.

So risk isn’t just time-varying; it’s also **downside-sensitive**.”

---

## Slide/Page 16 — VaR Backtesting (validate the volatility model)
“After fitting volatility, we validate it using a **Value-at-Risk (VaR) backtest**.

For a 5% VaR model, we expect about 5% of outcomes to breach the VaR threshold.

We observe **58 exceedances out of 1,300 observations**, which is **4.46%**—close to the expected level.

Interpretation:
- this supports that the volatility model is reasonably calibrated,
- and we can use it as a risk overlay in the later strategy.”

---

## Slide/Page 17 — Market Efficiency Results (multi-test evidence)
“Now we test whether KSE-30 behaves like a random walk or if predictability exists.

We use multiple tests because each test measures a different ‘randomness’ property.

The headline points:
- **Runs test:** borderline (p around **0.0561**) → sign randomness is close to efficient but not strongly rejected.
- **Variance Ratio tests:** support efficiency at horizons like VR(2) and VR(4).
- **Ljung–Box Q:** strongly indicates serial dependence (p ≈ **0.0000**) → contradicts pure random walk.
- **Hurst exponent:** **H ≈ 0.6559**, indicating persistence/long memory.

So the overall conclusion is **mixed efficiency**: not fully efficient and not fully inefficient.”

---

## Slide/Page 18–19 — Efficiency takeaway + bridge to rebalancing
“This mixed-efficiency result is crucial.

Because the market is not purely random, it is plausible that:
- there is exploitable structure,
- and forecasting/tilting strategies can be justified.

But since evidence is mixed, we avoid overclaiming. Instead, we position our work as:
- a probabilistic, decision-support framework
- that combines flow direction, volatility risk, and rebalancing inclusion/weight predictions.”

---

## One-sentence bridge
“From here, we move into portfolio rebalancing—predicting retention probability and weight changes at the constituent level.”

