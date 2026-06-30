# Stage Reference

This file is a short companion to `README.md`.

Use it when you want a quick stage-by-stage reminder without the installation, demo, config, or formula sections.

## `utils`

Purpose:
Load the dataset, normalize the cohort column, replace configured special values with missing values, and identify source features.

Outputs:
- `artifacts.data`
- `artifacts.clean_data`
- `artifacts.source_features`
- `artifacts.vars_over_preproc`

## `univariate`

Purpose:
Check each feature independently for missingness, special values, uniqueness, and obvious quality problems.

Outputs:
- `artifacts.num_report`
- `artifacts.cat_report`
- `artifacts.vars_over_miss`
- `artifacts.candidate_features`

## `psi`

Purpose:
Measure drift between the reference cohort and the other cohorts.

Outputs:
- `artifacts.psi_report`
- `artifacts.vars_over_psi`
- updated `artifacts.candidate_features`

## `bivariate`

Purpose:
Measure predictive strength against the binary target.

Outputs:
- `artifacts.bivariate_report`
- `artifacts.vars_over_gini`
- updated `artifacts.candidate_features`

## `missing`

Purpose:
Learn numeric imputations and categorical WoE mappings.

Outputs:
- `artifacts.impute_values`
- `artifacts.woe_mappings`

## `correlation_filter`

Purpose:
Reduce redundancy among remaining features.

Outputs:
- `artifacts.correlation_report`
- `artifacts.correlated_vars_dict`
- `artifacts.selected_features`

## `transform`

Purpose:
Apply the learned rules and build the final dataset.

Outputs:
- `artifacts.transformed_data`
