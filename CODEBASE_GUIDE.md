# Codebase Guide

This file is for contributors who want to understand how the package is organized before changing code.

## What This Package Does

`preprocessing_pipeline` is a small, stage-based preprocessing library for tabular binary-classification datasets.

At a high level, it:

1. reads a dataset
2. identifies valid feature candidates
3. removes weak, unstable, or redundant features
4. learns imputation and WoE rules
5. builds a transformed output dataset
6. exports reports that explain the decisions

The design is intentionally simple:

- one main pipeline class
- one shared artifacts object
- one module with stage implementations
- small helper modules for metrics, IO, config normalization, and reporting

## File Map

### Public entrypoints

- `__init__.py`
  Exposes the package-level import surface.

- `library.py`
  Home of `PreprocessingPipeline`, the main object most users interact with.

- `runner.py`
  CLI entrypoint. Loads a config file, runs the pipeline, and optionally exports reports.

### Core pipeline internals

- `config.py`
  Defines the canonical stage order and normalizes user config into a predictable shape.

- `artifacts.py`
  Defines `PipelineArtifacts`, the shared state container passed implicitly across stages through the pipeline object.

- `stages.py`
  Contains the actual stage implementations and the helper functions most tightly coupled to stage logic.

- `metrics.py`
  Holds reusable mathematical helpers such as PSI, Gini, WoE, special-value masking, and feature typing.

### IO and reporting

- `io_utils.py`
  Reads input datasets and writes output datasets.

- `reports.py`
  Exports CSV reports and a simple HTML report.

### Docs and examples

- `README.md`
  Main user guide.

- `STAGES.md`
  Short stage-by-stage reference.

- `NOTEBOOK_USAGE.md`
  Notebook-specific usage examples.

- `example_config.py`
  Example config structure.

- `demo/`
  Small runnable example with a toy dataset and demo runner.

## Execution Flow

The normal execution path is:

1. user builds a `PreprocessingPipeline(config)`
2. `config.py` normalizes the config
3. `library.py` creates an empty `PipelineArtifacts`
4. `run()` walks through the stage list in order
5. each stage in `stages.py` reads from and writes to `pipeline.artifacts`
6. if the run finishes at `transform`, the final dataset can be written to disk
7. reports can be exported separately through `get_report_csv()` and `get_report_html()`

## Stage Order

The canonical stage order lives in `config.py`:

1. `utils`
2. `univariate`
3. `psi`
4. `bivariate`
5. `missing`
6. `correlation_filter`
7. `transform`

This order is important because later stages depend on artifacts created earlier.

## Stage Responsibilities

### `utils`

Main job:

- load the input data
- normalize the cohort column to string
- replace configured special values with nulls
- define the initial source feature list

Main outputs written into `PipelineArtifacts`:

- `data`
- `clean_data`
- `source_features`
- `vars_over_preproc`

### `univariate`

Main job:

- inspect each feature independently
- measure missingness, special-value usage, and uniqueness
- remove obviously bad features early

Main outputs:

- `num_report`
- `cat_report`
- `vars_over_miss`
- `candidate_features`

### `psi`

Main job:

- compare each evaluation cohort against the reference cohort
- remove unstable features

Main outputs:

- `psi_report`
- `vars_over_psi`
- updated `candidate_features`

### `bivariate`

Main job:

- measure feature usefulness against the binary target
- remove weak features using the configured Gini threshold

Main outputs:

- `bivariate_report`
- `vars_over_gini`
- updated `candidate_features`

### `missing`

Main job:

- learn imputation values for surviving features
- learn WoE mappings for categorical variables

Main outputs:

- `impute_values`
- `woe_mappings`

### `correlation_filter`

Main job:

- transform current candidates into a comparable numeric frame
- compute pairwise correlations
- drop one variable from highly correlated pairs

Main outputs:

- `correlation_report`
- `correlated_vars_dict`
- `selected_features`

### `transform`

Main job:

- apply imputations
- apply WoE mappings to categorical features
- keep final selected features
- append configured non-feature columns

Main outputs:

- `transformed_data`

## Shared State Model

The pipeline uses a shared mutable state object instead of passing many intermediate return values between stages.

That object is `PipelineArtifacts` in `artifacts.py`.

This keeps the call signature simple, but it also means contributors should be careful about:

- stage ordering
- overwriting artifact fields
- partial runs, where some fields may still be empty

When changing a stage, always check which earlier artifacts it expects and which later stages depend on what it writes.

## Where To Make Common Changes

### Add a new stage

Update all of these places:

1. `config.py`
   Add the new stage name to `DEFAULT_STAGE_ORDER`.

2. `stages.py`
   Implement the stage function and add its explanation to `STAGE_EXPLANATIONS`.

3. `library.py`
   Register the stage in `self.stage_handlers`.

4. docs
   Update `README.md` and `STAGES.md`.

### Change filtering thresholds or config behavior

Start in `config.py`.

That is where default thresholds are defined and where config normalization happens.

### Change how a metric is computed

Start in `metrics.py`.

This is the best place for reusable math and scoring logic that should stay separate from orchestration.

### Change report export behavior

Start in `reports.py`.

### Change file-format support

Start in `io_utils.py`.

## Design Notes

A few choices are deliberate:

- categorical WoE mapping is learned in `missing`, not in `transform`
- correlation filtering happens after imputation rules are known so features can be compared numerically
- `runner.py` accepts Python config files to keep configuration flexible and easy to version
- reports are exported separately from `run()` so library users can choose when to write files

## Demo And Test Surface

There is no dedicated automated test suite yet.

For now, the fastest regression check is:

1. install the package in a clean environment
2. run `python demo/run_demo.py`
3. run `python runner.py --config demo/demo_config.py --reports-dir demo/demo_reports`
4. confirm reports and output dataset are created

If you add functionality, it is a good idea to verify both the Python API path and the CLI path.

## Good First Reading Order

If you are new to the project, the fastest way to build context is:

1. `README.md`
2. `library.py`
3. `config.py`
4. `artifacts.py`
5. `stages.py`
6. `metrics.py`
7. `runner.py`

That sequence goes from user-facing behavior to orchestration to internal logic.
