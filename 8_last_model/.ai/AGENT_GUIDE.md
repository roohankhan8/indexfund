# .ai Agent Guidance for 8_last_model

- Current primary objective: KSE-30 next-rebalance stay/exclusion + weight-change prediction.
- Primary script: `kse30_rebalance_pipeline.py`.
- Keep code self-contained and path-relative within `8_last_model`.
- Use NAV/AUM/flow from AKD, NBP, NTI as explicit features.
- Do not edit files under `data/`; treat as source inputs.
- Save all artifacts under `output/` only.
- If feature engineering or target definitions change, update `mds/8_last_model_workflow.md`.
- Keep deterministic seeds (`random_state=42`) for reproducible report results.
