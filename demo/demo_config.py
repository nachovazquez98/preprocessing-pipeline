from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent

config = {
    "path_data": str(DEMO_DIR / "sample_train.csv"),
    "path_df_output": str(DEMO_DIR / "sample_train_preprocessed.csv"),
    "target_column": "target",
    "cohort_column": "cohort_month",
    "reference_cohort": "202401",
    "non_features_list": [
        "customer_id",
        "target",
        "cohort_month",
    ],
    "categorical_variables": [
        "sector",
        "product_type",
        "signup_channel",
        "region_cluster",
    ],
    "special_values": [-999],
    "dictionary_special_values": {},
    "thresholds": {
        "max_p_missing": 0.35,
        "max_p_special": 0.35,
        "max_psi": 0.20,
        "min_gini": 0.30,
        "max_corr": 0.99,
    },
    "psi_bins": 5,
}
