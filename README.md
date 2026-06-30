# Open-Source Preprocessing Pipeline

`preprocessing_pipeline` is a reusable preprocessing package for tabular binary-classification datasets.

It helps you take a raw table, decide which columns are worth keeping, apply a small set of transformation rules, and produce:

- a transformed dataset
- feature-level reports
- learned imputations
- learned WoE mappings
- a final selected feature list

In practical terms, it is a screening and preparation tool for modeling data. It checks the health of your columns, removes risky ones, standardizes the surviving ones, and records why those decisions were made.

This README assumes your terminal is already inside `preprocessing_pipeline/`.

## What It Does

The package answers six questions in order:

1. Which columns are valid candidate features?
2. Which columns have too many missing or special values?
3. Which columns drift too much across cohorts?
4. Which columns are too weak against the binary target?
5. Which columns are redundant because another feature already carries the same information?
6. How should the surviving columns be imputed and encoded for modeling?

The result is a cleaner, smaller, more consistent dataset plus the reports needed to understand how it was created.

## Where To Go Next

Depending on what you want to do:

- use the package: continue with `Quick Start`
- understand the stages in more detail: see `STAGES.md`
- use it from a notebook: see `NOTEBOOK_USAGE.md`
- understand how the code is organized: see `CODEBASE_GUIDE.md`
- contribute to the project: see `CONTRIBUTING.md`

## Install

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -e .
```

Verify the environment:

```bash
python -c "import pandas, numpy, sklearn; print('environment ready')"
```

This installs the package plus its runtime dependencies.

Main dependencies:

- `pandas` for dataframe loading, cleaning, filtering, and reporting
- `numpy` for numeric helpers used in PSI and transformations
- `scikit-learn` for ROC AUC used to derive Gini
- `pyarrow` for parquet support

## Quick Start

The easiest path is:

1. run the demo
2. inspect the generated outputs
3. copy the demo config
4. replace the demo paths and column names with your own
5. rerun with your own config

### Run the demo

```bash
python demo/run_demo.py
```

Or with the generic CLI entrypoint:

```bash
python runner.py --config demo/demo_config.py --reports-dir demo/demo_reports
```

The demo generates:

- `demo/sample_train_preprocessed.csv`
- `demo/demo_reports/*.csv`
- `demo/demo_reports/preprocessing-report.html`
- `demo/demo_reports/ml_demo_metrics.csv`
- `demo/demo_reports/ml_demo_coefficients.csv`

This demo is not just a smoke test. It is built to show the full project philosophy in a compact scenario:

- a constant feature is removed early
- a high-missing feature is removed in univariate screening
- a special-value-heavy feature is removed in univariate screening
- a drifting cohort feature is removed by PSI
- a weak feature is removed in bivariate screening
- a redundant feature pair is resolved by the correlation filter
- the final dataset keeps a mix of numerical, categorical, and binary features
- the final transformed dataset can be used immediately in a small ML example

## Use It With Your Own Dataset

Copy the demo config:

```bash
cp demo/demo_config.py my_config.py
```

Your config file must expose a variable named:

- `config`
- or `EXAMPLE_CONFIG`

Then update these fields:

- `path_data`
- `path_df_output`
- `target_column`
- `cohort_column`
- `reference_cohort`
- `non_features_list`
- `categorical_variables`
- `special_values`
- `dictionary_special_values`
- `thresholds`
- `psi_bins`

Run your personalized config:

```bash
python runner.py --config my_config.py --reports-dir my_reports
```

## Pipeline Stages

The pipeline runs these stages in order:

1. `utils`
2. `univariate`
3. `psi`
4. `bivariate`
5. `missing`
6. `correlation_filter`
7. `transform`

### `utils`

Loads the dataset, standardizes the cohort column, replaces configured special values with missing values, and identifies source features.

Outputs:

- `artifacts.data`
- `artifacts.clean_data`
- `artifacts.source_features`
- `artifacts.preprocessing_exclusions`

### `univariate`

Checks each column independently for missingness, special-value usage, uniqueness, and obvious quality problems.

Outputs:

- `artifacts.numeric_report`
- `artifacts.categorical_report`
- `artifacts.univariate_exclusions`
- `artifacts.candidate_features`

### `psi`

Measures whether a feature behaves consistently across cohorts and removes unstable variables.

Outputs:

- `artifacts.psi_report`
- `artifacts.psi_exclusions`
- updated `artifacts.candidate_features`

### `bivariate`

Measures whether a feature is useful for predicting the binary target and removes weak variables.

Outputs:

- `artifacts.bivariate_report`
- `artifacts.bivariate_selected_features`
- updated `artifacts.candidate_features`

### `missing`

Learns how to fill missing values and how to convert categorical values into numeric form.

Outputs:

- `artifacts.impute_values`
- `artifacts.woe_mappings`

### `correlation_filter`

Finds pairs of features that overlap too much and keeps the stronger one.

Outputs:

- `artifacts.correlation_report`
- `artifacts.correlated_feature_decisions`
- `artifacts.selected_features`

### `transform`

Applies the learned rules and builds the final dataset.

Outputs:

- `artifacts.transformed_data`

For a shorter per-stage reference, see `STAGES.md`.

## What It Computes Automatically

The package makes its filtering decisions with a small set of formulas and thresholds.

### Missing rate

This measures how much of a column is empty.

`missing_rate = missing_count / n_rows`

If `missing_rate > max_p_missing`, the feature can be removed.

### Special-value rate

This measures how often a column uses placeholders such as `-999`.

`special_rate = special_count / n_rows`

If `special_rate > max_p_special`, the feature can be removed.

### PSI

This measures how much a feature changes from the reference cohort to later cohorts.

`PSI = sum((eval_i - ref_i) * ln(eval_i / ref_i))`

Where:

- `ref_i` is the bucket share in the reference cohort
- `eval_i` is the bucket share in the evaluation cohort

If `PSI > max_psi`, the feature can be removed.

### Gini

This measures how useful a feature is for separating the two target classes.

`Gini = 2 * AUC - 1`

If `Gini < min_gini`, the feature can be removed.

### WoE

This converts category labels into numbers in a way that reflects their relationship to the target.

`WoE(category) = ln(% good in category / % bad in category)`

### Correlation filter

This removes redundant variables after transformation.

`abs(correlation) > max_corr`

If two variables are too correlated, the one with the lower Gini is removed.

## Outputs

After a successful run, the most important outputs are:

- `artifacts.numeric_report`
- `artifacts.categorical_report`
- `artifacts.psi_report`
- `artifacts.bivariate_report`
- `artifacts.correlation_report`
- `artifacts.impute_values`
- `artifacts.woe_mappings`
- `artifacts.selected_features`
- `artifacts.transformed_data`

These outputs answer different questions:

- the reports explain what happened at each filtering stage
- the mappings explain how the final transformation was learned
- the selected feature list shows what survived
- the transformed dataset is the final output for downstream modeling

## Config Reference

Start from `example_config.py`.

Important fields:

- `path_data`: input dataset path
- `path_df_output`: output dataset path
- `target_column`: binary target column
- `cohort_column`: cohort/date column used by PSI
- `reference_cohort`: baseline cohort for stability comparison
- `non_features_list`: columns to preserve but not treat as candidate features
- `categorical_variables`: columns forced to behave as categorical
- `special_values`: global placeholders such as `-999`
- `dictionary_special_values`: column-specific placeholders
- `thresholds`: filtering thresholds
- `psi_bins`: number of bins for numeric PSI

Threshold keys:

- `max_p_missing`
- `max_p_special`
- `max_psi`
- `min_gini`
- `max_corr`

## Python Usage

After installing with `pip install -e .`, run the pipeline like a normal library:

```python
from preprocessing_pipeline import PreprocessingPipeline
from preprocessing_pipeline.runner import load_config_from_file

config = load_config_from_file("demo/demo_config.py")
pipeline = PreprocessingPipeline(config)
artifacts = pipeline.run()

print(artifacts.selected_features)
print(artifacts.transformed_data.head())
```

Generate reports:

```python
pipeline.get_report_csv(base_path="reports")
pipeline.get_report_html(base_path="reports")
```

## Notebook Usage

For Jupyter usage, see:

- `NOTEBOOK_USAGE.md`

That file contains a ready-to-use notebook cell with automatic project-root discovery and both config-file and DataFrame examples.

## CLI Usage

Run the demo config:

```bash
python runner.py --config demo/demo_config.py
```

Run with reports:

```bash
python runner.py --config demo/demo_config.py --reports-dir demo/demo_reports
```

Run only part of the pipeline:

```bash
python runner.py --config demo/demo_config.py --start utils --end bivariate
```

## Key Files

- `library.py`: main `PreprocessingPipeline` class
- `stages.py`: stage-by-stage execution logic
- `metrics.py`: PSI, Gini, WoE, and helper calculations
- `artifacts.py`: pipeline state and outputs
- `config.py`: config normalization and defaults
- `io_utils.py`: file readers and writers
- `reports.py`: CSV and HTML report generation
- `runner.py`: CLI entrypoint
- `example_config.py`: config template
- `STAGES.md`: short stage reference
- `NOTEBOOK_USAGE.md`: notebook-specific usage example
- `CODEBASE_GUIDE.md`: contributor-oriented codebase map
- `CONTRIBUTING.md`: setup, workflow, and contribution expectations
- `pyproject.toml`: editable-install package configuration

## Notes

- target must be binary
- PSI is cohort-based
- categorical features are transformed with WoE
- correlation filtering is pairwise and greedy
- parquet IO depends on `pyarrow`
