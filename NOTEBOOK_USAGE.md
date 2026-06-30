# Notebook Usage

This file gives you a ready-to-use notebook example for `preprocessing_pipeline`.

Use it when you want to:

- test the package inside Jupyter
- inspect reports interactively
- run the demo config without hardcoded absolute paths
- run the pipeline directly from a pandas DataFrame

Before using this notebook example, install the package from inside `preprocessing_pipeline/`:

```bash
pip install -e .
```

## Single Notebook Cell

Copy this cell into a notebook:

```python
from pathlib import Path
import pandas as pd

from preprocessing_pipeline import PreprocessingPipeline
from preprocessing_pipeline.runner import load_config_from_file
import preprocessing_pipeline

PACKAGE_DIR = Path(preprocessing_pipeline.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEMO_DIR = PACKAGE_DIR / "demo"

# Option 1: run with the demo config
config = load_config_from_file(str(DEMO_DIR / "demo_config.py"))
pipeline = PreprocessingPipeline(config)
artifacts = pipeline.run()

print("Selected features:", artifacts.selected_features)
display(artifacts.transformed_data.head())
display(artifacts.num_report.head())
display(artifacts.psi_report.head())
display(artifacts.bivariate_report.head())
display(artifacts.correlation_report.head())

# Optional: export reports
pipeline.get_report_csv(base_path="reports")
pipeline.get_report_html(base_path="reports")

# Option 2: run directly from a DataFrame
df = pd.read_csv(DEMO_DIR / "sample_train.csv")

config_df = {
    "path_data": "",
    "path_df_output": str(PROJECT_ROOT / "sample_train_preprocessed.csv"),
    "target_column": "target",
    "cohort_column": "join_date",
    "reference_cohort": "202301",
    "non_features_list": ["customer_id", "target", "join_date"],
    "nominal_variables": ["sector_str", "product_type_str"],
    "special_values": [],
    "dictionary_special_values": {},
    "thresholds": {
        "max_p_missing": 0.95,
        "max_p_special": 0.95,
        "max_psi": 0.50,
        "min_gini": 0.00,
        "max_corr": 0.85,
    },
    "psi_bins": 5,
}

pipeline_df = PreprocessingPipeline(config_df)
artifacts_df = pipeline_df.run(data=df)

print("Selected features from DataFrame run:", artifacts_df.selected_features)
display(artifacts_df.transformed_data.head())
```

## What This Does

- imports the installed package directly
- finds the package folder automatically
- runs the demo config
- shows the most important output reports
- shows how to run the pipeline directly from a pandas DataFrame

## When To Use Which Option

- Use the demo config option when you want the fastest working example.
- Use the DataFrame option when your notebook already has a dataset loaded.
