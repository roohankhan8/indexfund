# Chapter 6 Graph Explanations

## Figure 6.1
Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/C6_01_rebalancing_framework.png`

- This diagram summarizes the stock-level rebalancing workflow from constituent history to pre-event feature windows and finally to predicted weight and retention outcomes.
- Use it to explain that the rebalancing block is an application layer built on top of the earlier market and flow analysis.

## Figure 6.2
Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R02_feature_importances.png`

- This figure shows which rebalancing predictors contribute most strongly to the retained model family.
- Use it to support the discussion that current weight, size-related measures, and stability features matter more than short-run momentum alone.

## Figure 6.3
Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R03_weight_scatter.png`

- This scatter plot compares observed and predicted constituent weights in the test sample.
- Use it to show that weight persistence is strong and that the preferred model tracks larger weights reasonably well.
- The best weight model retained in the results file is Ridge with test R-squared of 0.9678.

## Figure 6.4
Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R01_retention_probability.png`

- This figure shows predicted retention probabilities across stocks in the test or forecast setting.
- Use it to explain why AUC is more informative than raw accuracy in a heavily imbalanced retention problem.
- The best retained inclusion model is Logistic with AUC of 0.8214.

## Figure 6.5
Path: `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R04_weight_changes.png`

- This graph shows predicted weight changes across the forecast universe.
- Use it to identify likely gainers, likely decliners, and names with near-zero target weight.
