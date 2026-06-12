# Chapter 3 Graph Explanations

## Figure 3.1
Path: `report_workspace/chapter-03-methodology/images/C3_01_methodological_progression.png`

- This diagram shows how the final methodology evolved from `4_claude_model/` to `5_claude_pipeline/` and finally to `6_cursor_model/`.
- Use it to explain that the dissertation does not treat the project as one sudden final script. It presents a clear methodological progression.
- The main interpretation is that folder 4 contributed modular research logic, folder 5 consolidated the workflow, and folder 6 became the final KSE-30 specific implementation.

## Figure 3.2
Path: `report_workspace/chapter-03-methodology/images/C3_02_final_data_flow.png`

- This figure shows the data pipeline from raw stock, fund, macro, and CPI inputs into the cleaned master datasets and then into the final KSE-30 analytics.
- Use it when describing how folder 5 prepares `daily_master.csv`, `monthly_master.csv`, and `kse30_stocks_clean.csv`, while folder 6 consumes those outputs.
- The key message is that preprocessing and final analytics are separated but connected in one reproducible workflow.

## Figure 3.3
Path: `report_workspace/chapter-03-methodology/images/C3_03_variable_families.png`

- This diagram groups the study variables into four families: target flow variables, macroeconomic drivers, KSE-30 market variables, and rebalancing features.
- Use it to show that the final methodology is not only a forecasting exercise. It combines fund, market, macro, and stock-level signals in one design.
- The interpretation should emphasize that each variable family supports a different analytical block of the project.
