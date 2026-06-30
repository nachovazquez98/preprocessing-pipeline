from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import (
    build_feature_score,
    build_special_mask,
    compute_gini,
    compute_psi,
    compute_woe_mapping,
    get_column_special_values,
    is_categorical_feature,
    nanmax_or_nan,
    prepare_binary_target,
)


STAGE_EXPLANATIONS = {
    "utils": "Reads data, normalizes cohort values, replaces special values with nulls, and identifies usable source features.",
    "univariate": "Builds per-feature quality checks such as missing rate, special-value rate, unique counts, and early feature removals.",
    "psi": "Measures population drift by comparing each cohort against the reference cohort using PSI and removes unstable variables.",
    "bivariate": "Measures predictive signal against the binary target using a simple open-source Gini approximation and keeps stronger variables.",
    "missing": "Learns imputation values and WoE mappings needed to convert features into a model-ready representation.",
    "correlation_filter": "Removes redundant variables by comparing pairwise correlations and keeping the higher-Gini feature.",
    "transform": "Applies imputations and WoE transformations, keeps the selected features, and writes the final dataset if an output path is configured.",
}


def get_source_features(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    excluded = set(config["non_features_list"])
    excluded.add(config["target_column"])
    excluded.add(config["cohort_column"])
    return [column for column in df.columns if column not in excluded]


def get_preprocessing_removals(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    removals = []
    for feature in get_source_features(df, config):
        if df[feature].dropna().nunique() <= 1:
            removals.append(feature)
    return removals


def build_univariate_row(df: pd.DataFrame, raw_df: pd.DataFrame, feature: str, config: Dict[str, Any]) -> Dict[str, Any]:
    series = df[feature]
    is_cat = is_categorical_feature(series, feature, config["nominal_variables"])
    special_mask = build_special_mask(raw_df[feature], feature, config)
    non_null_unique = series.dropna().nunique()
    p_missing = float(series.isna().mean())
    p_special = float(special_mask.mean())
    recommended_action = "keep"

    if non_null_unique <= 1:
        recommended_action = "remove"
    if p_missing >= config["thresholds"]["max_p_missing"]:
        recommended_action = "remove"
    if p_special >= config["thresholds"]["max_p_special"]:
        recommended_action = "remove"

    return {
        "name": feature,
        "dtype": "categorical" if is_cat else "numerical",
        "is_categorical": is_cat,
        "n_missing": int(series.isna().sum()),
        "p_missing": p_missing,
        "n_special": int(special_mask.sum()),
        "p_special": p_special,
        "n_unique": int(non_null_unique),
        "recommended_action": recommended_action,
    }


def report_removals(report: pd.DataFrame) -> List[str]:
    if report.empty:
        return []
    return report.loc[report["recommended_action"] == "remove", "name"].tolist()


def transform_feature_frame(clean_df: pd.DataFrame, features: List[str], artifacts, config: Dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(index=clean_df.index)
    for feature in features:
        series = clean_df[feature]
        if is_categorical_feature(series, feature, config["nominal_variables"]):
            fill_value = artifacts.impute_values.get(feature, "__missing__")
            mapping = artifacts.woe_mappings.get(feature, {})
            transformed = series.fillna(fill_value).astype(str)
            frame[feature] = transformed.map(mapping).fillna(0.0).astype(float)
        else:
            fill_value = artifacts.impute_values.get(feature, 0.0)
            frame[feature] = series.astype(float).fillna(fill_value)
    return frame


def run_utils_stage(pipeline, data: Optional[pd.DataFrame] = None):
    df = data.copy() if data is not None else pipeline._read_input_data()
    cohort_col = pipeline.config["cohort_column"]
    if cohort_col not in df.columns:
        raise ValueError(f"Missing cohort column '{cohort_col}' in input data.")
    df = df.copy()
    df[cohort_col] = df[cohort_col].astype(str)

    clean_df = df.copy()
    for column in clean_df.columns:
        specials = get_column_special_values(column, pipeline.config)
        if specials:
            clean_df.loc[clean_df[column].isin(specials), column] = pd.NA

    pipeline.artifacts.data = df
    pipeline.artifacts.clean_data = clean_df
    pipeline.artifacts.source_features = get_source_features(clean_df, pipeline.config)
    pipeline.artifacts.vars_over_preproc = get_preprocessing_removals(clean_df, pipeline.config)


def run_univariate_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    cohort_col = pipeline.config["cohort_column"]
    reference = pipeline.config["reference_cohort"]
    cohorts = sorted(clean_df[cohort_col].dropna().astype(str).unique().tolist())
    pipeline.artifacts.eval_cohorts = [value for value in cohorts if value != reference]

    num_rows = []
    cat_rows = []
    for feature in pipeline.artifacts.source_features:
        row = build_univariate_row(clean_df, pipeline.artifacts.data, feature, pipeline.config)
        if row["is_categorical"]:
            cat_rows.append(row)
        else:
            num_rows.append(row)

    pipeline.artifacts.num_report = pd.DataFrame(num_rows)
    pipeline.artifacts.cat_report = pd.DataFrame(cat_rows)
    over_num = report_removals(pipeline.artifacts.num_report)
    over_cat = report_removals(pipeline.artifacts.cat_report)
    pipeline.artifacts.vars_over_miss = sorted(
        set(over_num + over_cat + pipeline.artifacts.vars_over_preproc)
    )
    pipeline.artifacts.candidate_features = [
        feature
        for feature in pipeline.artifacts.source_features
        if feature not in pipeline.artifacts.vars_over_miss
    ]


def run_psi_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    cohort_col = pipeline.config["cohort_column"]
    reference = pipeline.config["reference_cohort"]
    ref_mask = clean_df[cohort_col].astype(str) == reference
    if ref_mask.sum() == 0:
        raise ValueError(f"Reference cohort '{reference}' has no rows.")

    results = []
    vars_over_psi = {}
    for feature in list(pipeline.artifacts.candidate_features):
        is_cat = is_categorical_feature(clean_df[feature], feature, pipeline.config["nominal_variables"])
        row = {"name": feature}
        ref_series = clean_df.loc[ref_mask, feature]
        for eval_cohort in pipeline.artifacts.eval_cohorts:
            eval_mask = clean_df[cohort_col].astype(str) == eval_cohort
            eval_series = clean_df.loc[eval_mask, feature]
            psi_value = compute_psi(ref_series, eval_series, is_cat, pipeline.config["psi_bins"])
            row[eval_cohort] = psi_value
            if pd.notna(psi_value) and psi_value >= pipeline.config["thresholds"]["max_psi"]:
                vars_over_psi.setdefault(feature, {})[eval_cohort] = psi_value
        row["max_value"] = nanmax_or_nan([row.get(cohort, float("nan")) for cohort in pipeline.artifacts.eval_cohorts])
        results.append(row)

    pipeline.artifacts.psi_report = pd.DataFrame(results)
    pipeline.artifacts.vars_over_psi = vars_over_psi
    pipeline.artifacts.candidate_features = [
        feature for feature in pipeline.artifacts.candidate_features if feature not in vars_over_psi
    ]


def run_bivariate_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    target = prepare_binary_target(clean_df[pipeline.config["target_column"]], pipeline.config["target_column"])
    rows = []
    vars_over_gini = {}
    min_gini = pipeline.config["thresholds"]["min_gini"]

    for feature in pipeline.artifacts.candidate_features:
        series = clean_df[feature]
        is_cat = is_categorical_feature(series, feature, pipeline.config["nominal_variables"])
        score = build_feature_score(series, target, is_cat)
        gini = compute_gini(target, score)
        action = "keep" if pd.notna(gini) and gini >= min_gini else "remove"
        row = {
            "name": feature,
            "dtype": "categorical" if is_cat else "numerical",
            "gini": gini,
            "recommended_action": action,
            "iv": pd.NA,
            "groups": int(series.nunique(dropna=True)),
        }
        rows.append(row)
        if action == "keep":
            vars_over_gini[feature] = row

    pipeline.artifacts.bivariate_report = pd.DataFrame(rows)
    pipeline.artifacts.vars_over_gini = vars_over_gini
    pipeline.artifacts.candidate_features = list(vars_over_gini.keys())


def run_missing_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    target = prepare_binary_target(clean_df[pipeline.config["target_column"]], pipeline.config["target_column"])
    impute_values = {}
    woe_mappings = {}

    for feature in pipeline.artifacts.candidate_features:
        series = clean_df[feature]
        is_cat = is_categorical_feature(series, feature, pipeline.config["nominal_variables"])
        if is_cat:
            non_null = series.dropna()
            impute_values[feature] = non_null.mode().iloc[0] if not non_null.empty else "__missing__"
            woe_mappings[feature] = compute_woe_mapping(series, target)
        else:
            impute_values[feature] = float(series.median()) if series.notna().any() else 0.0

    pipeline.artifacts.impute_values = impute_values
    pipeline.artifacts.woe_mappings = woe_mappings


def run_correlation_filter_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    transformed = transform_feature_frame(clean_df, pipeline.artifacts.candidate_features, pipeline.artifacts, pipeline.config)
    if transformed.empty:
        pipeline.artifacts.selected_features = []
        pipeline.artifacts.correlation_report = pd.DataFrame()
        return

    corr = transformed.corr(numeric_only=True)
    rows = []
    removed = set()
    gini_lookup = {row["name"]: row["gini"] for _, row in pipeline.artifacts.bivariate_report.iterrows()}

    columns = list(corr.columns)
    for idx, var_1 in enumerate(columns):
        for var_2 in columns[idx + 1:]:
            corr_value = corr.loc[var_1, var_2]
            gini_1 = gini_lookup.get(var_1, float("nan"))
            gini_2 = gini_lookup.get(var_2, float("nan"))
            rows.append(
                {
                    "var_1": var_1,
                    "var_2": var_2,
                    "Correlacion": corr_value,
                    "gini_var_1": gini_1,
                    "gini_var_2": gini_2,
                }
            )
            if pd.notna(corr_value) and abs(corr_value) > pipeline.config["thresholds"]["max_corr"]:
                if var_1 in removed or var_2 in removed:
                    continue
                removed_var = var_2 if gini_1 >= gini_2 else var_1
                kept_var = var_1 if removed_var == var_2 else var_2
                removed.add(removed_var)
                pipeline.artifacts.correlated_vars_dict[removed_var] = {
                    "Correlation_filter": True,
                    "Status": f"Removed {removed_var} due to correlation with {kept_var}",
                }

    pipeline.artifacts.correlation_report = pd.DataFrame(rows)
    pipeline.artifacts.selected_features = [
        feature for feature in pipeline.artifacts.candidate_features if feature not in removed
    ]


def run_transform_stage(pipeline):
    clean_df = pipeline._require_dataframe(pipeline.artifacts.clean_data, "clean_data")
    selected_features = list(pipeline.artifacts.selected_features or pipeline.artifacts.candidate_features)
    transformed = transform_feature_frame(clean_df, selected_features, pipeline.artifacts, pipeline.config)
    keep_columns = [col for col in pipeline.config["non_features_list"] if col in clean_df.columns]
    base = clean_df[keep_columns].copy() if keep_columns else pd.DataFrame(index=clean_df.index)
    pipeline.artifacts.transformed_data = pd.concat([base, transformed], axis=1)
