EXAMPLE_CONFIG = {
    "path_data": "data/train.csv",
    "path_df_output": "data/train_preprocessed.csv",
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
    ],
    "special_values": [-999, -9999],
    "dictionary_special_values": {
        "employment_tenure_months": [-1],
        "bureau_score_raw": [-99999],
    },
    "thresholds": {
        "max_p_missing": 0.35,
        "max_p_special": 0.35,
        "max_psi": 0.25,
        "min_gini": 0.30,
        "max_corr": 0.90,
    },
    "psi_bins": 5,
}
