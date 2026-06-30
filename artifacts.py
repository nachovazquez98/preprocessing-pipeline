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
    evaluation_cohorts: List[str] = field(default_factory=list)
    numeric_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    categorical_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    psi_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    bivariate_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocessing_exclusions: List[str] = field(default_factory=list)
    univariate_exclusions: List[str] = field(default_factory=list)
    psi_exclusions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bivariate_selected_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlated_feature_decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    impute_values: Dict[str, Any] = field(default_factory=dict)
    woe_mappings: Dict[str, Dict[Any, float]] = field(default_factory=dict)
