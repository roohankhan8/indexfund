**Discrepancies and Issues in the FYP Report**

**# Major Structural and Formatting Issues**
- **Inconsistent Appendix Labeling**: Appendix C is missing in the Table of Contents (jumps from B to D). TOC lists Appendix A, B, D, E.
- **Duplicate/Redundant Content**: List of Abbreviations has repeated entries (e.g., MAE listed twice, KPSS). List of Symbols and Abbreviations have overlaps and inconsistencies in formatting.
- **Figure/Table Numbering and References**: Many figures (e.g., Figure 3.1–3.5 in Methodology) are described but may not exist in the provided text or appendices. Cross-references in chapters point to non-existent or misnumbered items in appendices.
- **Date Inconsistencies**: Signatures dated "12 June 2026"; report copyright "June 2026". Project spans Aug 2025–June 2026. Current real-world context (as of tool use) makes some 2025/2026 citations plausible but many appear forward-projected or fabricated for the exercise.
- **TOC vs. Content Mismatch**: Chapter 2 Literature Review has subsections up to 2.9, but content flow has gaps. Similar minor mismatches in other chapters.

**# Content and Factual Discrepancies**
- **Sample Size and Data Issues**: Daily dataset "1,300 observations"; monthly "60 observations". In Appendix A, monthly fund-flow mentions "39 aligned monthly observations after differencing" — direct contradiction. Small monthly sample heavily limits ARIMAX/VAR reliability claims.
- **Model Performance Claims**: Negative R² values reported for ARIMAX/VAR (e.g., -0.1944), yet touted as "significant improvements." This is statistically misleading for out-of-sample forecasting in finance reports. Directional accuracy (75%) is highlighted, but magnitude prediction is poor.
- **Literature Review Problems**:
  - Numerous citations dated 2025–2026 (e.g., Ishtiaq Khan 2026, Zahid Iqbal 2025, Ubaida Fatima 2025 — advisor name coincidence?). These appear fabricated or AI-generated, raising plagiarism/originality concerns.
  - Heavy reliance on recent Pakistani KSE studies; limited global comparison or critical synthesis.
- **Market Efficiency Conclusions**: "Mixed" results (Runs/VR support efficiency; Ljung-Box/Hurst reject). Interpretation is balanced but overstates "pockets of inefficiency" for practical rebalancing without robust backtesting.
- **Contribution Statement**: Roohan Khan did "technical" work; others "theoretical." Common in group projects but should be more quantified (e.g., specific models coded).
- **SDGs**: Only two checked, but alignment with "Decent Work..." and "Industry, Innovation..." is superficial.

**# Methodology and Technical Weaknesses**
- **Data Sources**: Vague ("PSX, Yahoo Finance..."). No detailed cleaning logs, handling of survivorship bias in KSE-30 constituents, or exact rebalancing dates.
- **Stationarity/Tests**: ADF/KPSS mentioned, but limited details on lag selection, robustness.
- **No Transaction Costs/Robustness**: Rebalancing application lacks costs, slippage, or out-of-sample backtesting — critical flaw for "practical utility" claims.
- **Hybrid Models**: Promoted but results emphasize simpler ARIMAX/GARCH/Logistic. Overhyped ML (LSTM/RF) without strong evidence of superiority here.
- **Ethical/Limitations**: Generic; no discussion of data licensing, look-ahead bias risks, or model overfitting given small samples.

**# Writing and Presentation Issues**
- **Repetition**: Executive Summary, Introduction, and Literature overlap heavily.
- **Typos/Grammar**: Minor (e.g., "KKSE-30" in conclusion; formatting artifacts in text).
- **Visuals**: Many figures described but appendices reference pipeline outputs not fully integrated.
- **Plagiarism Risk**: High similarity to existing KSE literature; future-dated citations suspicious. Similarity index claims <20%, but needs verification.

**# Recommendations to Make the Report Perfect (Based on Best Practices)**

**General Structure (Harvard Economics Guide, GradCoach, etc.)**:
- Strong **Introduction** with clear research question/hypothesis.
- **Literature Review**: Critical synthesis, not summary. Identify gaps explicitly.
- **Methodology**: Reproducible (full code/data appendix, robustness checks).
- **Results**: Objective presentation; separate interpretation.
- **Discussion**: Link back to literature, limitations, policy implications.
- **Conclusion**: No new info; future directions actionable.

**Finance-Specific Improvements**:
- **Data Rigor**: Use longer samples if possible; validate KSE-30 reconstruction. Include transaction costs, walk-forward validation for rebalancing.
- **Models**: Emphasize explainability (SHAP for RF/Logistic), ensemble methods, bias checks. Compare more baselines.
- **Backtesting**: Full portfolio simulation with realistic constraints.
- **Visuals/Tables**: Professional, labeled, in appendices with captions.
- **Citations**: Use consistent style (APA/Chicago); verify all sources. Avoid future-dated or untraceable refs.
- **Originality**: Run full plagiarism check; document code/data pipeline (GitHub repo ideal).
- **Length/Polish**: Trim repetition; ensure consistent tense (past for completed work); professional formatting (consistent headings, page numbers).

**Actionable Steps**:
1. Fix all numbering/appendix inconsistencies.
2. Add robustness sections (sensitivity analysis, alternative specs).
3. Include code summary/full repo link in Appendix E.
4. Strengthen practical strategy with simulated performance metrics.
5. Update limitations to address small sample, data vintage, external shocks.
6. Get advisor/external review for citations and claims.

This MD file summarizes key issues. The report is ambitious and well-structured overall but undermined by data inconsistencies, overstated results, and citation red flags. Addressing these would elevate it significantly.