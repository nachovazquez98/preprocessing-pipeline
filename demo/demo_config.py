from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent

config = {
    "path_data": str(DEMO_DIR / "sample_train.csv"),
    "path_df_output": str(DEMO_DIR / "sample_train_preprocessed.csv"),
    "target_column": "target",
    "cohort_column": "join_date",
    "reference_cohort": "202301",
    "non_features_list": [
        "customer_id",
        "target",
        "join_date",
    ],
    "nominal_variables": [
        "sector_str",
        "product_type_str",
    ],
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
