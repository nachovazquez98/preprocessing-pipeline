from typing import Any, Dict


DEFAULT_STAGE_ORDER = [
    "utils",
    "univariate",
    "psi",
    "bivariate",
    "missing",
    "correlation_filter",
    "transform",
]


DEFAULT_THRESHOLDS = {
    "max_p_missing": 0.95,
    "max_p_special": 0.95,
    "max_psi": 0.25,
    "min_gini": 0.05,
    "max_corr": 0.60,
}


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    normalized["target_column"] = config.get("target_column", "target")
    normalized["cohort_column"] = config.get("cohort_column", "cohort")
    normalized["reference_cohort"] = str(config.get("reference_cohort", ""))
    normalized["path_data"] = config.get("path_data")
    normalized["path_df_output"] = config.get("path_df_output")
    normalized["non_features_list"] = list(config.get("non_features_list", []))
    normalized["categorical_variables"] = list(config.get("categorical_variables", []))
    normalized["special_values"] = list(config.get("special_values", []))
    normalized["dictionary_special_values"] = dict(config.get("dictionary_special_values", {}))

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(config.get("thresholds", {}))
    normalized["thresholds"] = thresholds
    normalized["psi_bins"] = int(config.get("psi_bins", 10))
    return normalized
