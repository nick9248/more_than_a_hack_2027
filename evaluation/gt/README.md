# Reference labels are not included

This directory preserves the layout of the team's `main` branch. Ground-truth CSVs and match metadata were intentionally removed from this review branch along with other data files. No label file is needed to run the model or validate its CSV format.

The original `evaluation/run_evaluation.py` is retained unchanged for explicitly supplied external references. Its default public-label file is absent here. Do not restore competition labels for model selection, tune on the public five-minute window, or review target prediction correctness. Use `evaluation/validate_submission.py` for the required format check; keep any authorized inputs, outputs and logs local and outside cloud AI contexts.
