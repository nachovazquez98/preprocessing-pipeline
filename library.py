from typing import Any, Dict, Optional

import pandas as pd

from .artifacts import PipelineArtifacts
from .config import DEFAULT_STAGE_ORDER, normalize_config
from .io_utils import read_input_data, write_output_data
from .reports import export_reports_csv, export_reports_html
from .stages import (
    STAGE_EXPLANATIONS,
    run_bivariate_stage,
    run_correlation_filter_stage,
    run_missing_stage,
    run_psi_stage,
    run_transform_stage,
    run_univariate_stage,
    run_utils_stage,
)


class PreprocessingPipeline:
    """
    Reusable open-source preprocessing pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = normalize_config(config)
        self.artifacts = PipelineArtifacts()
        self.stage_handlers = {
            "utils": run_utils_stage,
            "univariate": run_univariate_stage,
            "psi": run_psi_stage,
            "bivariate": run_bivariate_stage,
            "missing": run_missing_stage,
            "correlation_filter": run_correlation_filter_stage,
            "transform": run_transform_stage,
        }

    @classmethod
    def from_state(cls, config: Dict[str, Any], state: PipelineArtifacts):
        pipeline = cls(config)
        pipeline.artifacts = state
        return pipeline

    def available_stages(self):
        return list(DEFAULT_STAGE_ORDER)

    def describe_stages(self):
        return dict(STAGE_EXPLANATIONS)

    def state(self) -> PipelineArtifacts:
        return self.artifacts

    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        start: str = "utils",
        end: str = "transform",
    ) -> PipelineArtifacts:
        start_idx, end_idx = self._validate_stage_range(start, end)
        for stage in DEFAULT_STAGE_ORDER[start_idx:end_idx + 1]:
            if stage == "utils":
                self.stage_handlers[stage](self, data=data)
            else:
                self.stage_handlers[stage](self)

        if end == "transform" and self.artifacts.transformed_data is not None and self.config.get("path_df_output"):
            write_output_data(self.artifacts.transformed_data, self.config["path_df_output"])
        return self.artifacts

    def get_report_csv(self, report_name: str = "preprocessing-report", base_path: str = "./preprocessing-reports"):
        export_reports_csv(
            {
                "numeric_report": self.artifacts.numeric_report,
                "categorical_report": self.artifacts.categorical_report,
                "psi_report": self.artifacts.psi_report,
                "bivariate_report": self.artifacts.bivariate_report,
                "correlation_report": self.artifacts.correlation_report,
            },
            report_name=report_name,
            base_path=base_path,
        )

    def get_report_html(self, report_name: str = "preprocessing-report", base_path: str = "./preprocessing-reports"):
        summary = pd.DataFrame(
            {
                "metric": [
                    "source_features",
                    "candidate_features",
                    "selected_features",
                    "reference_cohort",
                ],
                "value": [
                    len(self.artifacts.source_features),
                    len(self.artifacts.candidate_features),
                    len(self.artifacts.selected_features),
                    self.config["reference_cohort"],
                ],
            }
        )
        export_reports_html(
            {
                "Numeric Report": self.artifacts.numeric_report,
                "Categorical Report": self.artifacts.categorical_report,
                "PSI Report": self.artifacts.psi_report,
                "Bivariate Report": self.artifacts.bivariate_report,
                "Correlation Report": self.artifacts.correlation_report,
            },
            summary=summary,
            report_name=report_name,
            base_path=base_path,
        )

    def _validate_stage_range(self, start: str, end: str):
        if start not in DEFAULT_STAGE_ORDER:
            raise ValueError(f"Unknown start stage '{start}'. Available stages: {DEFAULT_STAGE_ORDER}")
        if end not in DEFAULT_STAGE_ORDER:
            raise ValueError(f"Unknown end stage '{end}'. Available stages: {DEFAULT_STAGE_ORDER}")
        start_idx = DEFAULT_STAGE_ORDER.index(start)
        end_idx = DEFAULT_STAGE_ORDER.index(end)
        if start_idx > end_idx:
            raise ValueError(f"Invalid stage range: start '{start}' is after end '{end}'.")
        return start_idx, end_idx

    def _read_input_data(self) -> pd.DataFrame:
        path = self.config.get("path_data")
        if not path:
            raise ValueError("No input data provided. Pass a DataFrame to run() or set config['path_data'].")
        return read_input_data(path)

    def _require_dataframe(self, df: Optional[pd.DataFrame], name: str) -> pd.DataFrame:
        if df is None:
            raise ValueError(f"Pipeline state '{name}' is empty. Run the required previous stages first.")
        return df


def build_pipeline(config: Dict[str, Any]) -> PreprocessingPipeline:
    return PreprocessingPipeline(config)
