# Slides 14–19 — Additional Points to Consider (Speaker Notes / Backup)

Use these as optional “extra lines” if you have time, or as backup if an examiner asks “so what?”

---

## Slide 14 (Fund Flow Forecasting: Naive vs ARIMAX vs VAR) — Add-on points
- **What is being predicted?** Clarify: we forecast *net flow direction and episode timing*, not perfect PKR amounts.
- **Why naive fails:** RW ignores lagged macro/market reactions; if flows respond with delay, RW under-captures it.
- **Explain directional accuracy:**
  - If actual flow is positive and you predict positive → correct “regime”.
  - The model learns when inflows/outflows are more likely.
- **Negative R² (plain language):**
  - “Our point forecast misses the exact size, but that doesn’t mean the model is useless—direction is the decision lever.”
- **Analogy (optional):** weather-vane / traffic-light prediction: direction matters more than exact speed.

---

## Slide 15 (Fund Flow Plot/Table) — Add-on points
- **Why spikes appear:**
  - NAV moves → investor reaction via subscriptions/redemptions.
  - Fund reporting/timing effects can also create apparent jumps.
- **Key justification to say aloud:**
  - EDA shows weak same-time correlation → predictive signal likely comes from *lags*.
  - That’s why ARIMAX/VAR (lagged structure) outperforms RW.
- **How to interpret mismatch:**
  - “Even if the curve isn’t perfectly on top of reality, episodes align better.”
- **Bridge line:** “If we can time inflow/outflow episodes, we can tilt exposure and manage risk around them.”

---

## Slide 15–16 (Volatility Modeling: GARCH vs EGARCH) — Add-on points
- **Why we model volatility:** flows may be influenced by risk-on/risk-off regimes; risk drives investors’ behavior.
- **GARCH persistence (α+β):**
  - Interpret as “volatility memory”: once risk rises, it stays elevated for a while.
- **EGARCH leverage (gamma):**
  - Downside shocks amplify future volatility more than upside shocks.
- **Layman analogy:** ripples after a stone (persistence) + braking hurts stability more than acceleration.
- **Say it carefully:**
  - “We’re not claiming markets are predictable in returns, but the *risk level* is predictable via volatility dynamics.”

---

## Slide 16 (VaR Backtesting) — Add-on points
- **What is VaR in one sentence:**
  - “VaR is a threshold loss level we expect to be breached about X% of the time.”
- **Backtesting logic:**
  - Count breaches vs expected count.
- **Interpret your number (58 / 1300):**
  - “Observed ~4.46% vs nominal 5% → close enough to trust the risk calibration.”
- **Important nuance (exam-friendly):**
  - Calibration ≠ guarantee of profitability.
  - But it supports using the model as a risk overlay.

---

## Slide 17 (Market Efficiency: Multi-test evidence) — Add-on points
- **Explain “mixed efficiency” clearly:**
  - Different tests look at different “randomness” properties.
- **Runs test:**
  - Focus on randomness of signs (up/down order).
- **Variance ratio:**
  - Focus on how variance scales across horizons.
- **Ljung–Box:**
  - Focus on serial autocorrelation (predictability in time).
- **Hurst exponent:**
  - Long memory vs mean reversion.
- **Layman analogy (optional):**
  - Like rolling dice: sometimes it looks random, but if you test long enough you might see patterns.

---

## Slide 18–19 (Efficiency Takeaway → Bridge to Rebalancing) — Add-on points
- **What you should conclude (balanced):**
  - Evidence supports *some* exploitable structure, but not “free profit guaranteed.”
- **Why decision-support framing matters:**
  - Use probabilities and risk overlays instead of claiming deterministic trading signals.
- **Bridge to rebalancing (explicit):**
  - “We combine: flow direction (ARIMAX/VAR) + volatility/risk calibration (EGARCH/VaR) + constituent inclusion/weight predictions (logistic/ridge/RF).”
- **If asked “what is the strategy?”:**
  - “A probabilistic portfolio tilt around expected retention and risk-aware sizing.”

---

## Optional FAQ lines (if interrogated)
- **Q: Why not rely on R² for flow?**
  - “Because flows are noisy; sign/regime is more stable and more aligned with decision-making.”
- **Q: How does efficiency relate to your model?**
  - “Efficiency is mixed: if predictability were fully absent, the tilt framework would have little justification.”
- **Q: Does VaR backtesting prove the model works?**
  - “It validates calibration of risk thresholds, not profitability.”

