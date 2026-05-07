# Report Workspace

This folder is the main working area for the final FYP report text files and chapter-specific images.

## Structure

- `table-of-contents.txt`
- `session.md`
- `chapter-03-methodology/`
- `chapter-04-data-collection-and-processing/`
- `chapter-05-results-and-analysis/`
- `chapter-06-portfolio-tilt-and-rebalancing-application/`
- `chapter-07-discussion/`
- `chapter-08-conclusion-and-recommendations/`
- `graph_explanations/`

Each chapter folder contains:

- the chapter `.txt` draft
- an `images/` folder with copied figures relevant to that chapter

## Figure rule

- Use each graph only once in the report.
- If a later chapter needs a new visual, generate a chapter-specific one instead of reusing an earlier graph.
- Use `generate_additional_report_graphs.py` to create the custom Chapter 3 and Chapter 6 diagrams, sync the Chapter 5 and 6 figure copies, create the extra Chapter 4, 7, and 8 visuals, and rebuild the graph explanation notes.
- The pipelines now repair the isolated May 2024 zero-AUM data gap before plotting or modelling fund flows.

## References still kept elsewhere

- Sample report PDF: `0-docs/reports/FYDP.pdf`
- Current report PDF: `0-docs/reports/FYP Report (Analyzing Mutual Funds).pdf`
- Primary empirical outputs: `6_cursor_model/results_*.csv`

## Writing rule

Always follow the headings in `table-of-contents.txt` exactly when updating or adding report chapters.
