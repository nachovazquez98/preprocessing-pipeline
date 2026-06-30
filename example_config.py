EXAMPLE_CONFIG = {
    "path_data": "data/train.csv",
    "path_df_output": "data/train_preprocessed.csv",
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
    "special_values": [-999, -9999],
    "dictionary_special_values": {
        "age": [-1],
        "income": [-99999],
    },
    "thresholds": {
        "max_p_missing": 0.95,
        "max_p_special": 0.95,
        "max_psi": 0.25,
        "min_gini": 0.05,
        "max_corr": 0.60,
    },
    "psi_bins": 10,
}
