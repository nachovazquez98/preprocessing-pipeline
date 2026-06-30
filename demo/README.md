# Demo

This folder contains a tiny end-to-end demo for the open-source preprocessing pipeline.

Commands below assume your terminal is in `preprocessing_pipeline/`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Files

- `sample_train.csv`: small binary-classification dataset with three cohorts
- `demo_config.py`: runnable config file
- `run_demo.py`: helper script that runs the pipeline and exports reports

Run the demo:

```bash
python demo/run_demo.py
```

Or use the generic CLI runner:

```bash
python runner.py --config demo/demo_config.py --reports-dir demo/demo_reports
```

Outputs:

- `sample_train_preprocessed.csv`
- `demo_reports/*.csv`
- `demo_reports/preprocessing-report.html`

## Personalize It

To use this demo as a starting point for your own dataset:

1. Copy `demo_config.py`.
2. Change `path_data` to your dataset path.
3. Change `path_df_output` to your desired output file.
4. Change `target_column`, `cohort_column`, and `reference_cohort`.
5. Update `non_features_list`.
6. Update `nominal_variables`.
7. Update thresholds if you want different filtering strictness.
