# Demo

This folder contains a compact but more realistic end-to-end demo for the open-source preprocessing pipeline.

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

## What This Demo Is Designed To Show

The demo is intentionally structured so a user can see the main decisions the pipeline knows how to make:

- `single_value_flag`
  Removed during preprocessing because it carries no information.

- `employment_tenure_months`
  Removed in univariate screening because it has too much missingness.

- `bureau_score_raw`
  Removed in univariate screening because it uses the special placeholder value `-999` too often.

- `signup_channel`
  Removed in the PSI stage because its distribution changes heavily across cohorts.

- `newsletter_opt_in`
  Removed in bivariate screening because it is too weak against the target.

- `balance_avg_3m` and `balance_avg_6m`
  Deliberately made very similar so the correlation filter can show how it removes redundant information.

- `delinquency_flag`
  Kept as a binary feature.

- `sector` and `product_type`
  Kept as categorical features and transformed through WoE.

- `age`, `income`, and `utilization_ratio`
  Kept as numerical features.

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

The helper script also prints a short stage summary so you can quickly see:

- which features were excluded early
- which feature drifted
- which features failed bivariate screening
- which features survived into the final transformed dataset

## Personalize It

To use this demo as a starting point for your own dataset:

1. Copy `demo_config.py`.
2. Change `path_data` to your dataset path.
3. Change `path_df_output` to your desired output file.
4. Change `target_column`, `cohort_column`, and `reference_cohort`.
5. Update `non_features_list`.
6. Update `categorical_variables`.
7. Update thresholds if you want different filtering strictness.

## Suggested Reading Order

If you are new to the project, the easiest learning path is:

1. run `python demo/run_demo.py`
2. inspect the printed stage summary
3. open the CSV reports in `demo_reports/`
4. open `demo_config.py`
5. compare the original dataset with `sample_train_preprocessed.csv`

That sequence lets you see the package philosophy in action before you adapt it to your own data.
