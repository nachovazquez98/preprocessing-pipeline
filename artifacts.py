from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PipelineArtifacts:
    data: Optional[pd.DataFrame] = None
    clean_data: Optional[pd.DataFrame] = None
    transformed_data: Optional[pd.DataFrame] = None
    source_features: List[str] = field(default_factory=list)
    candidate_features: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    eval_cohorts: List[str] = field(default_factory=list)
    num_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    cat_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    psi_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    bivariate_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    vars_over_preproc: List[str] = field(default_factory=list)
    vars_over_miss: List[str] = field(default_factory=list)
    vars_over_psi: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vars_over_gini: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlated_vars_dict: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    impute_values: Dict[str, Any] = field(default_factory=dict)
    woe_mappings: Dict[str, Dict[Any, float]] = field(default_factory=dict)
