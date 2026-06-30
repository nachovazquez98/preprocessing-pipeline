import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def is_categorical_feature(series: pd.Series, feature: str, nominal_variables: List[str]) -> bool:
    if feature in nominal_variables:
        return True
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def get_column_special_values(feature: str, config: Dict[str, Any]) -> List[Any]:
    column_specials = list(config["dictionary_special_values"].get(feature, []))
    return list(dict.fromkeys(column_specials + config["special_values"]))


def build_special_mask(series: pd.Series, feature: str, config: Dict[str, Any]) -> pd.Series:
    specials = get_column_special_values(feature, config)
    if not specials:
        return pd.Series(False, index=series.index)
    return series.isin(specials).fillna(False)


def compute_psi(ref_series: pd.Series, eval_series: pd.Series, is_categorical: bool, psi_bins: int) -> float:
    if ref_series.dropna().empty or eval_series.dropna().empty:
        return np.nan

    if is_categorical or ref_series.dropna().nunique() <= 10:
        ref_buckets = ref_series.fillna("__missing__").astype(str)
        eval_buckets = eval_series.fillna("__missing__").astype(str)
    else:
        quantiles = np.linspace(0, 1, psi_bins + 1)
        bins = np.unique(np.nanquantile(ref_series.astype(float), quantiles))
        if len(bins) <= 2:
            ref_buckets = ref_series.fillna("__missing__").astype(str)
            eval_buckets = eval_series.fillna("__missing__").astype(str)
        else:
            ref_buckets = pd.cut(ref_series.astype(float), bins=bins, include_lowest=True, duplicates="drop").astype(str)
            eval_buckets = pd.cut(eval_series.astype(float), bins=bins, include_lowest=True, duplicates="drop").astype(str)
            ref_buckets = ref_buckets.fillna("__missing__")
            eval_buckets = eval_buckets.fillna("__missing__")

    all_buckets = sorted(set(ref_buckets.unique()).union(set(eval_buckets.unique())))
    ref_dist = ref_buckets.value_counts(normalize=True).reindex(all_buckets, fill_value=0.0)
    eval_dist = eval_buckets.value_counts(normalize=True).reindex(all_buckets, fill_value=0.0)
    epsilon = 1e-6
    ref_dist = ref_dist.clip(lower=epsilon)
    eval_dist = eval_dist.clip(lower=epsilon)
    return float(((eval_dist - ref_dist) * np.log(eval_dist / ref_dist)).sum())


def prepare_binary_target(target: pd.Series, target_column: str) -> pd.Series:
    non_null = target.dropna()
    unique_values = sorted(non_null.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Target column '{target_column}' must be binary. Found values: {unique_values}")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return target.map(mapping)


def build_feature_score(series: pd.Series, target: pd.Series, is_categorical: bool) -> pd.Series:
    if is_categorical:
        tmp = pd.DataFrame({"feature": series.fillna("__missing__"), "target": target})
        means = tmp.groupby("feature")["target"].mean()
        return tmp["feature"].map(means)
    filled = series.astype(float)
    return filled.fillna(filled.median() if filled.notna().any() else 0.0)


def compute_gini(target: pd.Series, score: pd.Series) -> float:
    valid = target.notna() & score.notna()
    if valid.sum() == 0 or score[valid].nunique() <= 1:
        return np.nan
    auc = roc_auc_score(target[valid], score[valid])
    auc = max(auc, 1 - auc)
    return float(2 * auc - 1)


def compute_woe_mapping(series: pd.Series, target: pd.Series) -> Dict[Any, float]:
    tmp = pd.DataFrame({"feature": series.fillna("__missing__"), "target": target})
    grouped = tmp.groupby("feature")["target"]
    total_bad = float((target == 1).sum())
    total_good = float((target == 0).sum())
    mapping = {}

    for category, values in grouped:
        bad = float((values == 1).sum()) + 0.5
        good = float((values == 0).sum()) + 0.5
        bad_rate = bad / (total_bad + 1.0)
        good_rate = good / (total_good + 1.0)
        mapping[category] = float(np.log(good_rate / bad_rate))
    return mapping


def nanmax_or_nan(values: List[float]) -> float:
    return float(np.nanmax(values)) if any(not math.isnan(value) for value in values) else np.nan
